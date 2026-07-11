use super::*;

pub struct Add_F32_1;
#[cube]
impl ReductionOp<f32_1> for Add_F32_1 {
    fn apply(x: f32_1, y: f32_1) -> f32_1 {
        x + y
    }
}

pub struct CellAve_F32_1;
#[cube]
impl UnaryOp<(f32, u32)> for CellAve_F32_1 {
    type Output = f32;

    fn apply(inp: (f32, u32)) -> f32 {
        let (v, n) = inp;
        f32_safe_div(v, n as f32)
    }
}

pub struct Add_F32_3;
#[cube]
impl ReductionOp<f32_3> for Add_F32_3 {
    fn apply(x: f32_3, y: f32_3) -> f32_3 {
        f32_3_add(x, y)
    }
}

pub struct Sub_F32_3;
#[cube]
impl UnaryOp<(f32_3, f32_3)> for Sub_F32_3 {
    type Output = f32_3;

    fn apply(x: (f32_3, f32_3)) -> f32_3 {
        f32_3_sub(x.0, x.1)
    }
}

pub struct CellAve_F32_3;
#[cube]
impl UnaryOp<(f32_3, u32)> for CellAve_F32_3 {
    type Output = f32_3;

    fn apply(x: (f32_3, u32)) -> f32_3 {
        let (u, v, w) = flatten3(x.0);
        let n = x.1;
        tuple3(
            f32_safe_div(u, n as f32),
            f32_safe_div(v, n as f32),
            f32_safe_div(w, n as f32),
        )
    }
}
