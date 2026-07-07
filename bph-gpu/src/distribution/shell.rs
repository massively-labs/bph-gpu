use super::*;

use std::f32::consts::PI;

pub fn alloc_shell_rand<R: Runtime>(
    exec: &Executor<R>,
    u: DeviceSliceMut<R, f32>,
    v: DeviceSliceMut<R, f32>,
    w: DeviceSliceMut<R, f32>,
    seed: u64,
) {
    let n = u.len();
    let uniform =
        massively::util::random::uniform_distribution_f32(exec, n * 2, 0., 1., seed).unwrap();
    let uniform1 = uniform.slice(0..n);
    let uniform2 = uniform.slice(n..2 * n);
    massively::transform(
        exec,
        Zip2(uniform1, uniform2),
        ShellRand,
        Zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )
    .unwrap();
}

struct ShellRand;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32)> for ShellRand {
    type Output = (f32, f32, f32);
    fn apply(x: (f32, f32)) -> (f32, f32, f32) {
        let (rand1, rand2) = x;
        let cs = 1. - 2. * rand1; // cs = [-1, 1)
        let sn = (1. - cs * cs).sqrt();
        let b = 2. * PI * rand2;

        let cx = sn * b.sin();
        let cy = sn * b.cos();
        let cz = cs;

        (cx, cy, cz)
    }
}
