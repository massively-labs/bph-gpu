use super::*;

/// IN:  idx=[0,0,2,2,2,2], k=3
/// OUT: [2,0,4]
pub fn bucket_counting<R: Runtime>(
    exec: &Executor<R>,
    idx: DeviceSlice<u32>,
    out: DeviceSliceMut<u32>,
) -> Result<(), massively::Error> {
    let k = out.len();
    let counting = massively::lazy::counting_u32(0).take(k);

    let begin = massively::vector::lower_bound(exec, idx.slice(..), counting, OrderingU32)?;

    let end = massively::vector::upper_bound(
        exec,
        idx.slice(..),
        massively::lazy::counting_u32(0).take(k),
        OrderingU32,
    )?;

    transform_into(exec, zip2(end.slice(..), begin.slice(..)), CalcDiff, out)
}

struct OrderingU32;
#[cube]
impl BinaryPredicateOp<u32> for OrderingU32 {
    fn apply(x: u32, y: u32) -> bool {
        x < y
    }
}

struct CalcDiff;
#[cube]
impl UnaryOp<(u32, u32)> for CalcDiff {
    type Output = u32;
    fn apply(x: (u32, u32)) -> u32 {
        let (end, begin) = x;
        end - begin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bucket_counting() {
        let exec = super::test_executor();
        let idx = exec.to_device(&[0_u32, 0, 2, 2, 2, 2]);
        let counts = exec.full(3, 0_u32).unwrap();

        bucket_counting(&exec, idx.slice(..), counts.slice_mut(..)).unwrap();

        assert_eq!(exec.to_host(&counts).unwrap(), vec![2, 0, 4]);
    }
}
