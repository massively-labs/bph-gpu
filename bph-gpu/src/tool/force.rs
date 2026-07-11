use super::*;

pub struct NoForce;
#[cube]
impl UnaryOp<f32_6> for NoForce {
    type Output = f32_3;
    fn apply(_: f32_6) -> Self::Output {
        tuple3(0_f32, 0_f32, 0_f32)
    }
}
