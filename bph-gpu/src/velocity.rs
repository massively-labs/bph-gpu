use super::*;

/// Subtracts the average velocity per cell and returns per-particle averages.
pub fn sub_average_velocity<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<R, f32>,
    v: DeviceSliceMut<R, f32>,
    w: DeviceSliceMut<R, f32>,
    idx: DeviceSlice<R, u32>,
    k: u32,
) -> (DeviceVec<R, f32>, DeviceVec<R, f32>, DeviceVec<R, f32>) {
    // Compute total velocity and particle count for each cell.
    let (SoA3(cell_sum_u, cell_sum_v, cell_sum_w), cell_cnt) = algorithm::reduce_by_bucket(
        exec,
        idx,
        SoA3(u.slice(..), v.slice(..), w.slice(..)),
        (0.0, 0.0, 0.0),
        common::Add_F32_3,
        k,
    );

    // Compute average velocity.
    let SoA3(cell_ave_u, cell_ave_v, cell_ave_w) = massively::map(
        exec,
        SoA4(
            cell_sum_u.slice(..),
            cell_sum_v.slice(..),
            cell_sum_w.slice(..),
            cell_cnt.slice(..),
        ),
        common::CellAve_F32_3,
        (),
    )
    .unwrap();

    // Subtract average velocity.
    let SoA3(ave_u, ave_v, ave_w) = massively::permute(
        exec,
        SoA3(
            cell_ave_u.slice(..),
            cell_ave_v.slice(..),
            cell_ave_w.slice(..),
        ),
        idx,
    )
    .unwrap();

    massively::transform(
        exec,
        SoA6(
            u.slice(..),
            v.slice(..),
            w.slice(..),
            ave_u.slice(..),
            ave_v.slice(..),
            ave_w.slice(..),
        ),
        common::Sub_F32_3,
        (),
        SoA3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )
    .unwrap();

    (ave_u, ave_v, ave_w)
}

struct AddAve;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32, f32, f32, f32)> for AddAve {
    type Env = ();
    type Output = (f32, f32, f32);

    fn apply(_env: (), x: (f32, f32, f32, f32, f32, f32)) -> (f32, f32, f32) {
        let (u, v, w, au, av, aw) = x;
        (u + au, v + av, w + aw)
    }
}

/// Adds previously subtracted average velocity back.
pub fn add_average_velocity<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<R, f32>,
    v: DeviceSliceMut<R, f32>,
    w: DeviceSliceMut<R, f32>,
    ave_u: DeviceSlice<R, f32>,
    ave_v: DeviceSlice<R, f32>,
    ave_w: DeviceSlice<R, f32>,
) {
    massively::transform(
        exec,
        SoA6(u.slice(..), v.slice(..), w.slice(..), ave_u, ave_v, ave_w),
        AddAve,
        (),
        SoA3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn velocity_case() -> impl Strategy<Value = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, u32)> {
        (Just(5000_usize), Just(1000_u32)).prop_flat_map(|(n, k)| {
            let component = prop::collection::vec(-1000.0_f32..1000.0, n);
            let idx = prop::collection::vec(0..k, n).prop_map(|mut idx| {
                idx.sort_unstable();
                idx
            });
            (component.clone(), component.clone(), component, idx)
                .prop_map(move |(u, v, w, idx)| (u, v, w, idx, k))
        })
    }

    fn prop_assert_f32s_close(actual: Vec<f32>, expected: &[f32]) -> Result<(), TestCaseError> {
        prop_assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let diff = (actual - expected).abs();
            let scale = expected.abs().max(1.0);
            prop_assert!(
                diff <= 1.0e-3 * scale,
                "actual={actual}, expected={expected}, diff={diff}"
            );
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn sub_then_add_average_velocity_restores_original(
            (host_u, host_v, host_w, host_idx, k) in velocity_case()
        ) {
            let exec = super::test_executor();
            let u = exec.to_device(&host_u).unwrap();
            let v = exec.to_device(&host_v).unwrap();
            let w = exec.to_device(&host_w).unwrap();
            let idx = exec.to_device(&host_idx).unwrap();

            let (ave_u, ave_v, ave_w) = sub_average_velocity(
                &exec,
                u.slice_mut(..),
                v.slice_mut(..),
                w.slice_mut(..),
                idx.slice(..),
                k,
            );

            add_average_velocity(
                &exec,
                u.slice_mut(..),
                v.slice_mut(..),
                w.slice_mut(..),
                ave_u.slice(..),
                ave_v.slice(..),
                ave_w.slice(..),
            );

            prop_assert_f32s_close(exec.to_host(&u).unwrap(), &host_u)?;
            prop_assert_f32s_close(exec.to_host(&v).unwrap(), &host_v)?;
            prop_assert_f32s_close(exec.to_host(&w).unwrap(), &host_w)?;
        }
    }
}
