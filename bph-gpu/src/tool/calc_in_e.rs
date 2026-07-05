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
    let Zip1(kietic_e) = exec.alloc::<(f32,)>(u.len()).unwrap();
    massively::transform(
        exec,
        Zip4(u, v, w, m),
        calc_kin_e::CalcKinE,
        (),
        Zip1(kietic_e.slice_mut(..)),
    )
    .unwrap();

    // Compute the kinetic energy sum for each cell.
    let Zip1(sum_kinetic_e) = Zip1(exec.constant(k, 0_f32).unwrap());
    let cnt = exec.constant(k, 0_u32).unwrap();
    algorithm::reduce_by_bucket(
        exec,
        idx.slice(..),
        Zip1(kietic_e.slice(..)),
        (0.,),
        common::Add_F32_1,
        Zip1(sum_kinetic_e.slice_mut(..)),
        cnt.slice_mut(..),
    )
    .unwrap();

    // Multiply by s/3 to compute the internal energy sum for each cell.
    let Zip1(sum_in_e) = exec.alloc::<(f32,)>(k).unwrap();
    massively::transform(
        exec,
        Zip1(sum_kinetic_e.slice(..)),
        CalcInE,
        s,
        Zip1(sum_in_e.slice_mut(..)),
    )
    .unwrap();

    // Divide by the particle count to get the per-particle internal energy for each cell.
    let Zip1(in_e) = exec.alloc::<(f32,)>(k).unwrap();
    massively::transform(
        exec,
        Zip2(sum_in_e.slice(..), cnt.slice(..)),
        common::CellAve_F32_1,
        (),
        Zip1(in_e.slice_mut(..)),
    )
    .unwrap();

    // Permute by particle index and return internal energy for each particle.
    let Zip1(out) = exec.alloc::<(f32,)>(idx.len()).unwrap();
    massively::gather(
        exec,
        Zip1(in_e.slice(..)),
        idx.slice(..),
        Zip1(out.slice_mut(..)),
    )
    .unwrap();

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
