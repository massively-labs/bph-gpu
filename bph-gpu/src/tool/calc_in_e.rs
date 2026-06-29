use super::*;

/// Computes internal energy from kinetic energy.
pub fn calc_in_e<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSlice<R, f32>,
    v: DeviceSlice<R, f32>,
    w: DeviceSlice<R, f32>,
    m: DeviceSlice<R, f32>,
    idx: DeviceSlice<R, u32>,
    k: u32,
    s: f32,
) -> DeviceVec<R, f32> {
    // Compute kinetic energy for each particle.
    let SoA1(kietic_e) = massively::map(exec, SoA4(u, v, w, m), calc_kin_e::CalcKinE, ()).unwrap();

    // Compute the kinetic energy sum for each cell.
    let (SoA1(sum_kinetic_e), cnt) = algorithm::reduce_by_bucket(
        exec,
        idx,
        SoA1(kietic_e.slice(..)),
        (0.,),
        common::Add_F32_1,
        k,
    );

    // Multiply by s/3 to compute the internal energy sum for each cell.
    let SoA1(sum_in_e) = massively::map(exec, SoA1(sum_kinetic_e.slice(..)), CalcInE, s).unwrap();

    // Divide by the particle count to get the per-particle internal energy for each cell.
    let SoA1(in_e) = massively::map(
        exec,
        SoA2(sum_in_e.slice(..), cnt.slice(..)),
        common::CellAve_F32_1,
        (),
    )
    .unwrap();

    // Permute by particle index and return internal energy for each particle.
    let SoA1(out) = massively::permute(exec, SoA1(in_e.slice(..)), idx).unwrap();

    out
}

struct CalcInE;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_1> for CalcInE {
    type Env = f32;
    type Output = f32_1;
    fn apply(env: Self::Env, inp: f32_1) -> f32_1 {
        let (x,) = inp;
        let s = env;
        (x * s / 3.,)
    }
}
