use super::*;

/// Computes internal energy from kinetic energy.
pub fn calc_in_e<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSlice<f32>,
    v: DeviceSlice<f32>,
    w: DeviceSlice<f32>,
    m: DeviceSlice<f32>,
    idx: DeviceSlice<u32>,
    k: u32,
    s: f32,
) -> DeviceVec<R, f32> {
    // Compute kinetic energy for each particle.
    let kinetic_e = exec.alloc::<f32>(u.len());
    massively::transform(
        exec,
        zip4(u, v, w, m),
        calc_kin_e::CalcKinE,
        kinetic_e.slice_mut(..),
    )
    .unwrap();

    // Compute the kinetic energy sum for each cell.
    let sum_kinetic_e = exec.full(k as usize, 0_f32).unwrap();
    let cnt = exec.full(k as usize, 0_u32).unwrap();
    algorithm::reduce_by_bucket(
        exec,
        idx.slice(..),
        kinetic_e.slice(..),
        0.,
        common::Add_F32_1,
        sum_kinetic_e.slice_mut(..),
        cnt.slice_mut(..),
    )
    .unwrap();

    // Multiply by s/3 to compute the internal energy sum for each cell.
    let sum_in_e = exec.alloc::<f32>(k as usize);
    massively::transform(
        exec,
        zip2(
            sum_kinetic_e.slice(..),
            massively::lazy::constant(s).take(sum_kinetic_e.len() as u32),
        ),
        CalcInE,
        sum_in_e.slice_mut(..),
    )
    .unwrap();

    // Divide by the particle count to get the per-particle internal energy for each cell.
    let in_e = exec.alloc::<f32>(k as usize);
    massively::transform(
        exec,
        zip2(sum_in_e.slice(..), cnt.slice(..)),
        common::CellAve_F32_1,
        in_e.slice_mut(..),
    )
    .unwrap();

    // Permute by particle index and return internal energy for each particle.
    let out = exec.alloc::<f32>(idx.len());
    massively::gather(exec, in_e.slice(..), idx.slice(..), out.slice_mut(..)).unwrap();

    out
}

struct CalcInE;
#[cube]
impl UnaryOp<f32_2> for CalcInE {
    type Output = f32_1;
    fn apply(inp: f32_2) -> f32_1 {
        let (x, s) = inp;
        x * s / 3.
    }
}
