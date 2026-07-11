use crate::tool::boundary::Range;

use super::*;

pub fn alloc_uniform_random<R: Runtime>(
    exec: &Executor<R>,
    x: DeviceSliceMut<f32>,
    range: Range,
    seed: u64,
) {
    let n = x.len();
    let xs = massively::util::random::uniform_f32(range.lo, range.hi, seed)
        .unwrap()
        .take(n as u32);
    massively::transform(exec, xs, massively::op::Identity, x).unwrap();
}
