use super::*;

/// IN:  idx=[0,0,2,2,2,2], k=3
/// OUT: [2,0,4]
pub fn bucket_counting<R: Runtime>(
    exec: &Executor<R>,
    idx: DeviceSlice<R, u32>,
    k: u32,
) -> DeviceVec<R, u32> {
    let counting = exec.counting(k).unwrap();

    let begin = massively::lower_bound(
        exec,
        SoA1(idx.slice(..)),
        SoA1(counting.slice(..)),
        OrderingU32,
    )
    .unwrap();

    let end = massively::upper_bound(
        exec,
        SoA1(idx.slice(..)),
        SoA1(counting.slice(..)),
        OrderingU32,
    )
    .unwrap();

    let SoA1(diff) =
        massively::map(exec, SoA2(end.slice(..), begin.slice(..)), CalcDiff, ()).unwrap();

    diff
}

struct OrderingU32;
#[cube]
impl<R: Runtime> BinaryPredicateOp<R, (u32,)> for OrderingU32 {
    fn apply(x: (u32,), y: (u32,)) -> bool {
        x.0 < y.0
    }
}

struct CalcDiff;
#[cube]
impl<R: Runtime> UnaryOp<R, (u32, u32)> for CalcDiff {
    type Env = ();
    type Output = (u32,);
    fn apply(_env: (), x: (u32, u32)) -> (u32,) {
        let (end, begin) = x;
        (end - begin,)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bucket_counting() {
        let exec = super::test_executor();
        let idx = exec.to_device(&[0_u32, 0, 2, 2, 2, 2]).unwrap();

        let counts = bucket_counting(&exec, idx.slice(..), 3);

        assert_eq!(exec.to_host(&counts).unwrap(), vec![2, 0, 4]);
    }
}
