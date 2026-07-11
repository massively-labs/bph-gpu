use super::*;

/// Computes kinetic energy.
/// (u,v,w,m) -> kinetic e
#[allow(dead_code)]
pub struct CalcKinE;
#[cube]
impl UnaryOp<f32_4> for CalcKinE {
    type Output = f32;
    fn apply(x: f32_4) -> f32 {
        let (u, v, w, m) = flatten4(x);
        0.5 * m * (u * u + v * v + w * w)
    }
}
