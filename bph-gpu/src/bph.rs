use super::*;

pub fn bph<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<f32>,
    v: DeviceSliceMut<f32>,
    w: DeviceSliceMut<f32>,
    m: DeviceSlice<f32>,       // Mass for each particle.
    in_e: DeviceSliceMut<f32>, // Internal energy.
    idx: DeviceSlice<u32>,     // Cell index for each particle.
    k: u32,                    // Number of cells.
    s: f32,                    // Degrees of freedom.
    seed: u64,
) {
    // Subtract average velocity.
    let (ave_u, ave_v, ave_w) = velocity::sub_average_velocity(
        exec,
        u.slice_mut(..),
        v.slice_mut(..),
        w.slice_mut(..),
        idx.slice(..),
        k,
    );

    // Relax velocities and internal energy.
    relax::relax(
        exec,
        u.slice_mut(..),
        v.slice_mut(..),
        w.slice_mut(..),
        m,
        in_e,
        idx.slice(..),
        k,
        s,
        seed,
    );

    // Add average velocity back.
    velocity::add_average_velocity(
        exec,
        u,
        v,
        w,
        ave_u.slice(..),
        ave_v.slice(..),
        ave_w.slice(..),
    );
}
