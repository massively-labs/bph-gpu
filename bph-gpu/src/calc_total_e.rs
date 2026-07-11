use super::*;

/// Computes total energy.
/// (u,v,w,m,in_e) -> total e
pub struct CalcTotalE;
#[cube]
impl UnaryOp<f32_5> for CalcTotalE {
    type Output = f32;
    // (u,v,w,m,in_e)
    fn apply(x: f32_5) -> f32 {
        let (u, v, w, m, in_e) = flatten5(x);
        let kin_e = 0.5 * m * (u * u + v * v + w * w);
        kin_e + in_e
    }
}
