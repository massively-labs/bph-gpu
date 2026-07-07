use super::*;

pub struct Add_F32_1;
#[cube]
impl<R: Runtime> ReductionOp<R, f32_1> for Add_F32_1 {
    fn apply(x: f32_1, y: f32_1) -> f32_1 {
        (x.0 + y.0,)
    }
}

pub struct CellAve_F32_1;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32, u32)> for CellAve_F32_1 {
    type Output = (f32,);

    fn apply(inp: (f32, u32)) -> (f32,) {
        let (v, n) = inp;
        (f32_safe_div(v, n as f32),)
    }
}

pub struct Add_F32_3;
#[cube]
impl<R: Runtime> ReductionOp<R, f32_3> for Add_F32_3 {
    fn apply(x: f32_3, y: f32_3) -> f32_3 {
        (x.0 + y.0, x.1 + y.1, x.2 + y.2)
    }
}

pub struct Sub_F32_3;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32_3, f32_3)> for Sub_F32_3 {
    type Output = f32_3;

    fn apply(x: (f32_3, f32_3)) -> (f32, f32, f32) {
        let (u, v, w) = x.0;
        let (au, av, aw) = x.1;
        (u - au, v - av, w - aw)
    }
}

pub struct CellAve_F32_3;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32_3, u32)> for CellAve_F32_3 {
    type Output = f32_3;

    fn apply(x: (f32_3, u32)) -> f32_3 {
        let (u, v, w) = x.0;
        let n = x.1;
        (
            f32_safe_div(u, n as f32),
            f32_safe_div(v, n as f32),
            f32_safe_div(w, n as f32),
        )
    }
}
