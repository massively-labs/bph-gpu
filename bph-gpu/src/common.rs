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
    type Env = ();
    type Output = (f32,);

    fn apply(_env: (), inp: (f32, u32)) -> (f32,) {
        let (v, n) = inp;
        (f32_safe_div(v, n as f32),)
    }
}

pub struct Add_F32_3;
#[cube]
impl<R: Runtime> ReductionOp<R, (f32, f32, f32)> for Add_F32_3 {
    fn apply(x: (f32, f32, f32), y: (f32, f32, f32)) -> (f32, f32, f32) {
        (x.0 + y.0, x.1 + y.1, x.2 + y.2)
    }
}

pub struct Sub_F32_3;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32, f32, f32, f32)> for Sub_F32_3 {
    type Env = ();
    type Output = (f32, f32, f32);

    fn apply(_env: (), x: (f32, f32, f32, f32, f32, f32)) -> (f32, f32, f32) {
        let (u, v, w, au, av, aw) = x;
        (u - au, v - av, w - aw)
    }
}

pub struct CellAve_F32_3;
#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32, u32)> for CellAve_F32_3 {
    type Env = ();
    type Output = (f32, f32, f32);

    fn apply(_env: (), x: (f32, f32, f32, u32)) -> (f32, f32, f32) {
        let (u, v, w, n) = x;
        (
            f32_safe_div(u, n as f32),
            f32_safe_div(v, n as f32),
            f32_safe_div(w, n as f32),
        )
    }
}
