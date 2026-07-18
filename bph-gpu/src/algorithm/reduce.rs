use super::*;

/// IN:  v=[1,2,3,4,5], idx=[0,0,2,2,2], k=3
/// OUT: ([3,0,12], [2,0,3])
pub fn reduce_by_bucket<R: Runtime, V, Sum, Output>(
    exec: &Executor<R>,
    idx: DeviceSlice<u32>,
    v: V,
    zero: Output::Item,
    sum: Sum,
    out: Output,
    counts: DeviceSliceMut<u32>,
) -> Result<(), massively::Error>
where
    Output: MIterMut<R>,
    V: MIter<R, Item = Output::Item>,
    Sum: ReductionOp<Output::Item>,
    Output::Item: massively::ToCanonical<R>
        + massively::WritableFrom<<Output::Item as massively::ToCanonical<R>>::Canonical>,
{
    let (cell_idx, scratch) =
        massively::vector::reduce_by_key(exec, idx.slice(..), v, Equal, zero, sum)?;

    massively::vector::scatter(
        exec,
        scratch.slice(..),
        massively::lazy::transform(cell_idx.slice(..), massively::op::U32ToUsize),
        out,
    )?;

    counting::bucket_counting(exec, idx, counts)
}

struct Equal;

#[cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumU32;

    #[cube]
    impl ReductionOp<u32> for SumU32 {
        fn apply(x: u32, y: u32) -> u32 {
            x + y
        }
    }

    #[test]
    fn test_reduce_by_bucket() {
        let exec = super::test_executor();
        let idx = exec.to_device(&[0_u32, 0, 2, 2, 2]);
        let v = exec.to_device(&[1_u32, 2, 3, 4, 5]);
        let sum = exec.full(3, 0_u32).unwrap();
        let counts = exec.full(3, 0_u32).unwrap();

        reduce_by_bucket(
            &exec,
            idx.slice(..),
            v.slice(..),
            0,
            SumU32,
            sum.slice_mut(..),
            counts.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&sum).unwrap(), vec![3, 0, 12]);
        assert_eq!(exec.to_host(&counts).unwrap(), vec![2, 0, 3]);
    }
}
