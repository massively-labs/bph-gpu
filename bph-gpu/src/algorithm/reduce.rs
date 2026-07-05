use super::*;

/// IN:  v=[1,2,3,4,5], idx=[0,0,2,2,2], k=3
/// OUT: ([3,0,12], [2,0,3])
pub fn reduce_by_bucket<R: Runtime, V, Sum, Output>(
    exec: &Executor<R>,
    idx: DeviceSlice<R, u32>,
    v: V,
    zero: Output::Item,
    sum: Sum,
    out: Output,
    counts: DeviceSliceMut<R, u32>,
) -> Result<(), massively::Error>
where
    Output: MIterMut<R>,
    V: MIter<R, Item = Output::Item>,
    Sum: ReductionOp<R, Output::Item>,
{
    let cell_idx = exec.alloc::<(u32,)>(idx.len())?;
    let cell_v = exec.alloc::<Output::Item>(massively::MIter::len(&v))?;
    let n = massively::reduce_by_key(
        exec,
        Zip1(idx.slice(..)),
        v,
        massively::op::Equal,
        zero,
        sum,
        cell_idx.slice_mut(..),
        cell_v.slice_mut(..),
    )?;

    massively::scatter(
        exec,
        cell_v.slice(..n),
        cell_idx.0.slice(..n),
        out,
    )?;

    counting::bucket_counting(exec, idx, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumU32;

    #[cube]
    impl<R: Runtime> ReductionOp<R, (u32,)> for SumU32 {
        fn apply(x: (u32,), y: (u32,)) -> (u32,) {
            (x.0 + y.0,)
        }
    }

    #[test]
    fn test_reduce_by_bucket() {
        let exec = super::test_executor();
        let idx = exec.to_device(&[0_u32, 0, 2, 2, 2]).unwrap();
        let v = exec.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
        let sum = Zip1(exec.constant(3, 0_u32).unwrap());
        let counts = exec.constant(3, 0_u32).unwrap();

        reduce_by_bucket(
            &exec,
            idx.slice(..),
            Zip1(v.slice(..)),
            (0,),
            SumU32,
            sum.slice_mut(..),
            counts.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&sum.0).unwrap(), vec![3, 0, 12]);
        assert_eq!(exec.to_host(&counts).unwrap(), vec![2, 0, 3]);
    }
}
