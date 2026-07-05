use super::*;

/// IN:  idx=[0,0,2,2,2,2], k=3
/// OUT: [2,0,4]
pub fn bucket_counting<R: Runtime>(
    exec: &Executor<R>,
    idx: DeviceSlice<R, u32>,
    out: DeviceSliceMut<R, u32>,
) -> Result<(), massively::Error> {
    let k = out.len();
    let counting = exec.counting(k)?;

    let begin = exec.constant(k, 0_u32)?;
    massively::lower_bound(
        exec,
        Zip1(idx.slice(..)),
        Zip1(counting.slice(..)),
        OrderingU32,
        begin.slice_mut(..),
    )?;

    let end = exec.constant(k, 0_u32)?;
    massively::upper_bound(
        exec,
        Zip1(idx.slice(..)),
        Zip1(counting.slice(..)),
        OrderingU32,
        end.slice_mut(..),
    )?;

    massively::transform(
        exec,
        Zip2(end.slice(..), begin.slice(..)),
        CalcDiff,
        (),
        Zip1(out),
    )
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
        let counts = exec.constant(3, 0_u32).unwrap();

        bucket_counting(&exec, idx.slice(..), counts.slice_mut(..)).unwrap();

        assert_eq!(exec.to_host(&counts).unwrap(), vec![2, 0, 4]);
    }
}
