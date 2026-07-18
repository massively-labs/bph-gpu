use super::*;

use std::f32::consts::PI;

pub fn alloc_shell_rand<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<f32>,
    v: DeviceSliceMut<f32>,
    w: DeviceSliceMut<f32>,
    seed: u64,
) {
    let n = u.len();
    let uniform1 = massively::util::random::uniform_f32(0., 1., seed)
        .unwrap()
        .take(n);
    let uniform2 = massively::util::random::uniform_f32(0., 1., seed.wrapping_add(1))
        .unwrap()
        .take(n);
    crate::algorithm::transform_into(
        exec,
        zip2(uniform1, uniform2),
        ShellRand,
        zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )
    .unwrap();
}

struct ShellRand;
#[cube]
impl UnaryOp<(f32, f32)> for ShellRand {
    type Output = f32_3;
    fn apply(x: (f32, f32)) -> f32_3 {
        let (rand1, rand2) = x;
        let cs = 1. - 2. * rand1; // cs = [-1, 1)
        let sn = (1. - cs * cs).sqrt();
        let b = 2. * PI * rand2;

        let cx = sn * b.sin();
        let cy = sn * b.cos();
        let cz = cs;

        tuple3(cx, cy, cz)
    }
}
