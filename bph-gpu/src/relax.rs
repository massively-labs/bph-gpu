use super::*;

pub fn alloc_balanced_shell_rand<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<f32>,
    v: DeviceSliceMut<f32>,
    w: DeviceSliceMut<f32>,
    idx: DeviceSlice<u32>,
    k: u32,
    seed: u64,
) {
    distribution::shell::alloc_shell_rand(
        exec,
        u.slice_mut(..),
        v.slice_mut(..),
        w.slice_mut(..),
        seed,
    );

    let cell_sum_u = exec.full(k as usize, 0_f32).unwrap();
    let cell_sum_v = exec.full(k as usize, 0_f32).unwrap();
    let cell_sum_w = exec.full(k as usize, 0_f32).unwrap();
    let cell_cnt = exec.full(k as usize, 0_u32).unwrap();
    algorithm::reduce::reduce_by_bucket(
        exec,
        idx.slice(..),
        zip3(u.slice(..), v.slice(..), w.slice(..)),
        tuple3(0., 0., 0.),
        common::Add_F32_3,
        zip3(
            cell_sum_u.slice_mut(..),
            cell_sum_v.slice_mut(..),
            cell_sum_w.slice_mut(..),
        ),
        cell_cnt.slice_mut(..),
    )
    .unwrap();

    let Zip(Zip(cell_ave_u, cell_ave_v), cell_ave_w) = exec.alloc::<f32_3>(k as usize);
    massively::transform(
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

    let ave = massively::lazy::permute(
        zip3(
            cell_ave_u.slice(..),
            cell_ave_v.slice(..),
            cell_ave_w.slice(..),
        ),
        idx.slice(..),
    );

    massively::transform(
        exec,
        zip2(zip3(u.slice(..), v.slice(..), w.slice(..)), ave),
        common::Sub_F32_3,
        zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )
    .unwrap();
}

pub fn relax<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<f32>,
    v: DeviceSliceMut<f32>,
    w: DeviceSliceMut<f32>,
    m: DeviceSlice<f32>,
    in_e: DeviceSliceMut<f32>,
    idx: DeviceSlice<u32>,
    k: u32,
    s: f32, // Degrees of freedom.
    seed: u64,
) {
    // -----------------------------------------------------------------
    // 1. Compute total energy and particle counts for each cell.
    // -----------------------------------------------------------------

    let total_e = exec.alloc::<f32>(idx.len());
    massively::transform(
        exec,
        zip5(
            u.slice(..),
            v.slice(..),
            w.slice(..),
            m.slice(..),
            in_e.slice(..),
        ),
        calc_total_e::CalcTotalE,
        total_e.slice_mut(..),
    )
    .unwrap();

    let cell_sum_total_e = exec.full(k as usize, 0_f32).unwrap();
    let cell_cnt = exec.full(k as usize, 0_u32).unwrap();
    algorithm::reduce::reduce_by_bucket(
        exec,
        idx.slice(..),
        total_e.slice(..),
        0.,
        common::Add_F32_1,
        cell_sum_total_e.slice_mut(..),
        cell_cnt.slice_mut(..),
    )
    .unwrap();

    // Relaxation models collisions by redistributing total energy into kinetic
    // and internal energy.
    // Cells with fewer than two particles cannot collide; redistributing their
    // energy would artificially lose energy.
    let collision_stencil = massively::lazy::transform(
        massively::lazy::permute(cell_cnt.slice(..), idx.slice(..)),
        IsCollidable,
    );

    // -----------------------------------------------------------------
    // 2. Assign shell-distributed velocities.
    // -----------------------------------------------------------------

    alloc_balanced_shell_rand(
        exec,
        u.slice_mut(..),
        v.slice_mut(..),
        w.slice_mut(..),
        idx.slice(..),
        k,
        seed,
    );

    // -----------------------------------------------------------------
    // 3. Assign new kinetic energy.
    // -----------------------------------------------------------------

    // 3.1 Compute current kinetic energy for each cell after shell assignment.
    let kinetic_e = exec.alloc::<f32>(idx.len());
    massively::transform(
        exec,
        zip4(u.slice(..), v.slice(..), w.slice(..), m.slice(..)),
        calc_kin_e::CalcKinE,
        kinetic_e.slice_mut(..),
    )
    .unwrap();

    let cell_sum_kin_e = exec.full(k as usize, 0_f32).unwrap();
    let cell_cnt_tmp = exec.full(k as usize, 0_u32).unwrap();
    algorithm::reduce_by_bucket(
        exec,
        idx.slice(..),
        kinetic_e.slice(..),
        0.,
        common::Add_F32_1,
        cell_sum_kin_e.slice_mut(..),
        cell_cnt_tmp.slice_mut(..),
    )
    .unwrap();

    // 3.2 Compute kinetic energy redistributed from total energy for each cell.
    // Target kinetic energy = total energy * 3 / (3+s).
    let cell_sum_tobe_kin_e = exec.alloc::<f32>(k as usize);
    massively::transform(
        exec,
        zip2(
            cell_sum_total_e.slice(..),
            massively::lazy::constant(s).take(cell_sum_total_e.len() as u32),
        ),
        DistributeKinE,
        cell_sum_tobe_kin_e.slice_mut(..),
    )
    .unwrap();

    // 3.3 Compute the velocity ratio from the kinetic energy ratio.
    // Ratio = sqrt(target kinetic energy / actual kinetic energy).
    let cell_vel_ratio = exec.alloc::<f32>(k as usize);
    massively::transform(
        exec,
        zip2(cell_sum_tobe_kin_e.slice(..), cell_sum_kin_e.slice(..)),
        CalcVelocityRatio,
        cell_vel_ratio.slice_mut(..),
    )
    .unwrap();

    // 3.4 Scale velocities by the ratio.
    massively::transform(
        exec,
        zip4(
            u.slice(..),
            v.slice(..),
            w.slice(..),
            massively::lazy::permute(cell_vel_ratio.slice(..), idx.slice(..)),
        ),
        ScaleVelocity,
        zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )
    .unwrap();

    // -----------------------------------------------------------------
    // 4. Assign new internal energy.
    // -----------------------------------------------------------------

    // Compute new internal energy.
    let cell_sum_tobe_in_e = exec.alloc::<f32>(k as usize);
    massively::transform(
        exec,
        zip2(
            cell_sum_total_e.slice(..),
            massively::lazy::constant(s).take(cell_sum_total_e.len() as u32),
        ),
        DistributeInE,
        cell_sum_tobe_in_e.slice_mut(..),
    )
    .unwrap();

    let cell_tobe_in_e = exec.alloc::<f32>(k as usize);
    massively::transform(
        exec,
        zip2(cell_sum_tobe_in_e.slice(..), cell_cnt.slice(..)),
        common::CellAve_F32_1,
        cell_tobe_in_e.slice_mut(..),
    )
    .unwrap();

    massively::gather_where(
        exec,
        cell_tobe_in_e.slice(..),
        idx.slice(..),
        collision_stencil.slice(..),
        in_e,
    )
    .unwrap();
}

struct IsCollidable;
#[cube]
impl UnaryOp<u32> for IsCollidable {
    type Output = u32;
    fn apply(inp: u32) -> u32 {
        if inp >= 2 { 1u32 } else { 0u32 }
    }
}

struct DistributeKinE;
#[cube]
impl UnaryOp<f32_2> for DistributeKinE {
    type Output = f32_1;
    fn apply(inp: f32_2) -> f32_1 {
        let (total_e, s) = inp;
        total_e * 3. / (3. + s)
    }
}

struct DistributeInE;
#[cube]
impl UnaryOp<f32_2> for DistributeInE {
    type Output = f32_1;
    fn apply(inp: f32_2) -> f32_1 {
        let (total_e, s) = inp;
        total_e * s / (3. + s)
    }
}

struct CalcVelocityRatio;
#[cube]
impl UnaryOp<f32_2> for CalcVelocityRatio {
    type Output = f32_1;
    fn apply(inp: f32_2) -> f32_1 {
        let (x, y) = inp;
        if y == 0. { 0_f32 } else { (x / y).sqrt() }
    }
}

struct ScaleVelocity;
#[cube]
impl UnaryOp<f32_4> for ScaleVelocity {
    type Output = f32_3;
    fn apply(inp: f32_4) -> f32_3 {
        let (u, v, w, ratio) = flatten4(inp);
        tuple3(u * ratio, v * ratio, w * ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Before entering relax, velocities in cells with fewer than two particles
    // should already be zero. Relaxation relies on that invariant.
    fn zero_single_particle_cell_velocity(
        u: &mut [f32],
        v: &mut [f32],
        w: &mut [f32],
        idx: &[u32],
        k: u32,
    ) {
        let mut cnt = vec![0_u32; k as usize];
        for &cell in idx {
            cnt[cell as usize] += 1;
        }

        for i in 0..idx.len() {
            if cnt[idx[i] as usize] < 2 {
                u[i] = 0.0;
                v[i] = 0.0;
                w[i] = 0.0;
            }
        }
    }

    fn relax_case() -> impl Strategy<Value = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, u32)>
    {
        (Just(5000_usize), Just(1000_u32)).prop_flat_map(|(n, k)| {
            let component = prop::collection::vec(-1000.0_f32..1000.0, n);
            let in_e = prop::collection::vec(0.0_f32..1.0e6, n);
            let idx = prop::collection::vec(0..k, n).prop_map(|mut idx| {
                idx.sort_unstable();
                idx
            });

            (component.clone(), component.clone(), component, in_e, idx).prop_map(
                move |(mut u, mut v, mut w, in_e, idx)| {
                    zero_single_particle_cell_velocity(&mut u, &mut v, &mut w, &idx, k);
                    (u, v, w, in_e, idx, k)
                },
            )
        })
    }

    fn total_energy(u: &[f32], v: &[f32], w: &[f32], m: &[f32], in_e: &[f32]) -> f64 {
        u.iter()
            .zip(v)
            .zip(w)
            .zip(m)
            .zip(in_e)
            .map(|((((&u, &v), &w), &m), &in_e)| {
                let u = u as f64;
                let v = v as f64;
                let w = w as f64;
                0.5 * m as f64 * (u * u + v * v + w * w) + in_e as f64
            })
            .sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]

        #[test]
        fn relax_preserves_total_energy(
            (host_u, host_v, host_w, host_in_e, host_idx, k) in relax_case()
        ) {
            let exec = super::test_executor();
            let host_m = vec![1.0_f32; host_u.len()];
            let before_total_e = total_energy(&host_u, &host_v, &host_w, &host_m, &host_in_e);
            let u = exec.to_device(&host_u);
            let v = exec.to_device(&host_v);
            let w = exec.to_device(&host_w);
            let m = exec.to_device(&host_m);
            let idx = exec.to_device(&host_idx);
            let s = 3.0_f32;
            let in_e = exec.to_device(&host_in_e);

            relax(
                &exec,
                u.slice_mut(..),
                v.slice_mut(..),
                w.slice_mut(..),
                m.slice(..),
                in_e.slice_mut(..),
                idx.slice(..),
                k,
                s,
                0,
            );

            let out_u = exec.to_host(&u).unwrap();
            let out_v = exec.to_host(&v).unwrap();
            let out_w = exec.to_host(&w).unwrap();
            let out_in_e = exec.to_host(&in_e).unwrap();

            prop_assert_eq!(out_u.len(), host_u.len());
            prop_assert_eq!(out_v.len(), host_v.len());
            prop_assert_eq!(out_w.len(), host_w.len());
            prop_assert_eq!(out_in_e.len(), host_idx.len());
            prop_assert!(out_u.iter().all(|x| x.is_finite()));
            prop_assert!(out_v.iter().all(|x| x.is_finite()));
            prop_assert!(out_w.iter().all(|x| x.is_finite()));
            prop_assert!(out_in_e.iter().all(|x| x.is_finite()));

            let after_total_e = total_energy(&out_u, &out_v, &out_w, &host_m, &out_in_e);
            // Total energy is conserved before and after relaxation.
            let diff = (after_total_e - before_total_e).abs();
            let tolerance = before_total_e.abs().max(1.0) * 1.0e-7;
            prop_assert!(
                diff <= tolerance,
                "before_total_e={before_total_e}, after_total_e={after_total_e}, diff={diff}, tolerance={tolerance}"
            );
        }
    }
}
