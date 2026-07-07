use super::*;

pub struct NoForce;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_6> for NoForce {
    type Output = f32_3;
    fn apply(_: f32_6) -> Self::Output {
        (0. as f32, 0. as f32, 0. as f32)
    }
}
