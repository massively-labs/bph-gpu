use super::*;

/// Computes kinetic energy.
/// (u,v,w,m) -> kinetic e
#[allow(dead_code)]
pub struct CalcKinE;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32, f32)> for CalcKinE {
    type Output = (f32,);
    fn apply(x: (f32, f32, f32, f32)) -> (f32,) {
        let (u, v, w, m) = x;
        (0.5 * m * (u * u + v * v + w * w),)
    }
}
