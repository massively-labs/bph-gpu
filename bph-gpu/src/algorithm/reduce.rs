use super::*;

/// IN:  v=[1,2,3,4,5], idx=[0,0,2,2,2], k=3
/// OUT: ([3,0,12], [2,0,3])
pub fn reduce_by_bucket<R: Runtime, V, Sum>(
    exec: &Executor<R>,
    idx: DeviceSlice<R, u32>,
    v: V,
    zero: V::Item,
    sum: Sum,
    k: u32,
) -> (<V::Item as MItem<R>>::Vec, DeviceVec<R, u32>)
where
    V: MIter<R>,
    Sum: ReductionOp<R, V::Item>,
{
    let (SoA1(cell_idx), cell_v) = massively::reduce_by_key(
        exec,
        SoA1(idx.slice(..)),
        v,
        massively::op::Equal,
        zero,
        sum,
    )
    .unwrap();

    let data = exec.alloc::<V::Item>(k).unwrap();
    massively::fill(exec, zero, data.slice_mut(..)).unwrap();
    massively::scatter(
        exec,
        cell_v.slice(..),
        cell_idx.slice(..),
        data.slice_mut(..),
    )
    .unwrap();

    let counting = counting::bucket_counting(exec, idx, k);

    (data, counting)
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

        let (SoA1(sum), counts) =
            reduce_by_bucket(&exec, idx.slice(..), SoA1(v.slice(..)), (0,), SumU32, 3);

        assert_eq!(exec.to_host(&sum).unwrap(), vec![3, 0, 12]);
        assert_eq!(exec.to_host(&counts).unwrap(), vec![2, 0, 3]);
    }
}
