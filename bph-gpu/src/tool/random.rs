use crate::tool::boundary::Range;

use super::*;

pub fn alloc_uniform_random<R: Runtime>(
    exec: &Executor<R>,
    x: DeviceSliceMut<R, f32>,
    range: Range,
    seed: u64,
) {
    let n = x.len();
    let xs = massively::util::random::uniform_distribution_f32(exec, n, range.lo, range.hi, seed)
        .unwrap();
    exec.copy(xs.slice(..), x).unwrap();
}
