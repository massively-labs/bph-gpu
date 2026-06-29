use super::*;

pub struct NoForce;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_7> for NoForce {
    type Env = ();
    type Output = f32_3;
    fn apply(_env: (), _: f32_7) -> Self::Output {
        (0. as f32, 0. as f32, 0. as f32)
    }
}
