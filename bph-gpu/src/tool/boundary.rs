use super::*;

pub type Wall = f32;

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone))]
pub struct Range {
    pub lo: f32,
    pub hi: f32,
}

pub struct OutHi;
#[cube]
impl UnaryOp<f32_2> for OutHi {
    type Output = bool;
    fn apply(inp: f32_2) -> bool {
        let (v, wall) = inp;
        v > wall
    }
}

pub struct OutLo;
#[cube]
impl UnaryOp<f32_2> for OutLo {
    type Output = bool;
    fn apply(inp: f32_2) -> bool {
        let (v, wall) = inp;
        v < wall
    }
}

pub struct ReflectHi;
#[cube]
impl UnaryOp<f32_2> for ReflectHi {
    type Output = f32_1;
    fn apply(inp: f32_2) -> f32_1 {
        let (x, wall) = inp;
        // x - (x-wall)*2
        wall * 2. - x
    }
}

pub struct ReflectLo;
#[cube]
impl UnaryOp<f32_2> for ReflectLo {
    type Output = f32_1;
    fn apply(inp: f32_2) -> f32_1 {
        let (x, wall) = inp;
        // x + (wall-x)*2
        wall * 2. - x
    }
}

pub struct Negate;
#[cube]
impl UnaryOp<f32_1> for Negate {
    type Output = f32_1;
    fn apply(inp: f32_1) -> f32_1 {
        -inp
    }
}

pub struct WrapHi;
#[cube]
impl UnaryOp<f32_3> for WrapHi {
    type Output = f32_1;
    fn apply(inp: f32_3) -> f32_1 {
        let (x, lo, hi) = flatten3(inp);
        x - (hi - lo)
    }
}

pub struct WrapLo;
#[cube]
impl UnaryOp<f32_3> for WrapLo {
    type Output = f32_1;
    fn apply(inp: f32_3) -> f32_1 {
        let (x, lo, hi) = flatten3(inp);
        x + (hi - lo)
    }
}
