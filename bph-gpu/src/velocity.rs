use super::*;

/// Subtracts the average velocity per cell and returns per-particle averages.
pub fn sub_average_velocity<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<f32>,
    v: DeviceSliceMut<f32>,
    w: DeviceSliceMut<f32>,
    idx: DeviceSlice<u32>,
    k: u32,
) -> (DeviceVec<R, f32>, DeviceVec<R, f32>, DeviceVec<R, f32>) {
    // Compute total velocity and particle count for each cell.
    let cell_sum_u = exec.full(k as usize, 0_f32).unwrap();
    let cell_sum_v = exec.full(k as usize, 0_f32).unwrap();
    let cell_sum_w = exec.full(k as usize, 0_f32).unwrap();
    let cell_cnt = exec.full(k as usize, 0_u32).unwrap();
    algorithm::reduce_by_bucket(
        exec,
        idx.slice(..),
        zip3(u.slice(..), v.slice(..), w.slice(..)),
        tuple3(0.0, 0.0, 0.0),
        common::Add_F32_3,
        zip3(
            cell_sum_u.slice_mut(..),
            cell_sum_v.slice_mut(..),
            cell_sum_w.slice_mut(..),
        ),
        cell_cnt.slice_mut(..),
    )
    .unwrap();

    // Compute average velocity.
    let Zip(Zip(cell_ave_u, cell_ave_v), cell_ave_w) = exec.alloc::<f32_3>(k as usize);
    crate::algorithm::transform_into(
        exec,
        zip2(
            zip3(
                cell_sum_u.slice(..),
                cell_sum_v.slice(..),
                cell_sum_w.slice(..),
            ),
            cell_cnt.slice(..),
        ),
        common::CellAve_F32_3,
        zip3(
            cell_ave_u.slice_mut(..),
            cell_ave_v.slice_mut(..),
            cell_ave_w.slice_mut(..),
        ),
    )
    .unwrap();

    // Subtract average velocity.
    let Zip(Zip(ave_u, ave_v), ave_w) = massively::vector::gather(
        exec,
        zip3(
            cell_ave_u.slice(..),
            cell_ave_v.slice(..),
            cell_ave_w.slice(..),
        ),
        massively::lazy::transform(idx.slice(..), massively::op::U32ToUsize),
    )
    .unwrap();

    crate::algorithm::transform_into(
        exec,
        zip2(
            zip3(u.slice(..), v.slice(..), w.slice(..)),
            zip3(ave_u.slice(..), ave_v.slice(..), ave_w.slice(..)),
        ),
        common::Sub_F32_3,
        zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )
    .unwrap();

    (ave_u, ave_v, ave_w)
}

struct AddAve;
#[cube]
impl UnaryOp<(f32_3, f32_3)> for AddAve {
    type Output = f32_3;

    fn apply(x: (f32_3, f32_3)) -> f32_3 {
        f32_3_add(x.0, x.1)
    }
}

/// Adds previously subtracted average velocity back.
pub fn add_average_velocity<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<f32>,
    v: DeviceSliceMut<f32>,
    w: DeviceSliceMut<f32>,
    ave_u: DeviceSlice<f32>,
    ave_v: DeviceSlice<f32>,
    ave_w: DeviceSlice<f32>,
) {
    crate::algorithm::transform_into(
        exec,
        zip2(
            zip3(u.slice(..), v.slice(..), w.slice(..)),
            zip3(ave_u, ave_v, ave_w),
        ),
        AddAve,
        zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
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
            let u = exec.to_device(&host_u);
            let v = exec.to_device(&host_v);
            let w = exec.to_device(&host_w);
            let idx = exec.to_device(&host_idx);

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
