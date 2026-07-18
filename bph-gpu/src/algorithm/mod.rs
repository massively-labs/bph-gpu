use super::*;

pub mod counting;
pub mod reduce;

pub use counting::bucket_counting;
pub use reduce::reduce_by_bucket;

/// Sorts particle values by `u32` cell key and returns the sorted keys as well.
///
/// `massively` 0.80 returns only the sorted values from `sort_by_key`.  BPH also
/// needs the sorted cell keys for its segmented reductions, so this helper
/// sorts an index permutation once and gathers both outputs with it.
pub fn sort_by_key_with_keys<R, Values, Less>(
    exec: &Executor<R>,
    keys: DeviceSlice<u32>,
    values: Values,
    less: Less,
) -> Result<(DeviceVec<R, u32>, massively::MVec<R, Values::Item>), massively::Error>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: massively::ToCanonical<R>,
    Less: BinaryPredicateOp<u32>,
{
    let permutation = massively::vector::sort_by_key(
        exec,
        keys.slice(..),
        massively::lazy::counting_u32(0).take(keys.len()),
        less,
    )?;
    let sorted_keys = massively::vector::gather(
        exec,
        keys,
        massively::lazy::transform(permutation.slice(..), massively::op::U32ToUsize),
    )?;
    let sorted_values = massively::vector::gather(
        exec,
        values,
        massively::lazy::transform(permutation.slice(..), massively::op::U32ToUsize),
    )?;
    Ok((sorted_keys, sorted_values))
}

/// Applies a transform into caller-provided storage using the public
/// `massively` 0.80 APIs.
pub fn transform_into<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), massively::Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item>,
    Op::Output: massively::ToCanonical<R>,
    Output::Item: massively::WritableFrom<<Op::Output as massively::ToCanonical<R>>::Canonical>,
{
    let len = input.len()?;
    let transformed = massively::vector::transform(exec, input, op)?;
    massively::vector::scatter(
        exec,
        transformed.slice(..),
        massively::lazy::counting(0).take(len),
        output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LessU32;

    #[cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> bool {
            lhs < rhs
        }
    }

    #[test]
    fn sort_by_key_keeps_all_columns_aligned() {
        let exec = crate::test_executor();
        let keys = exec.to_device(&[2_u32, 0, 2, 1]);
        let first = exec.to_device(&[20_u32, 0, 21, 10]);
        let second = exec.to_device(&[200_u32, 0, 210, 100]);

        let (sorted_keys, Zip(sorted_first, sorted_second)) = sort_by_key_with_keys(
            &exec,
            keys.slice(..),
            zip2(first.slice(..), second.slice(..)),
            LessU32,
        )
        .unwrap();

        assert_eq!(exec.to_host(&sorted_keys).unwrap(), vec![0, 1, 2, 2]);
        assert_eq!(exec.to_host(&sorted_first).unwrap(), vec![0, 10, 20, 21]);
        assert_eq!(
            exec.to_host(&sorted_second).unwrap(),
            vec![0, 100, 200, 210]
        );
    }
}
