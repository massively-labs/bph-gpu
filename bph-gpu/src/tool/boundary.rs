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
impl<R: Runtime> UnaryOp<R, f32_1> for OutHi {
    type Env = Wall;
    type Output = (u32,);
    fn apply(env: Wall, inp: f32_1) -> (u32,) {
        let v = inp.0;
        let wall = env;
        if v > wall { (1u32,) } else { (0u32,) }
    }
}

pub struct OutLo;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_1> for OutLo {
    type Env = Wall;
    type Output = (u32,);
    fn apply(env: Wall, inp: f32_1) -> (u32,) {
        let v = inp.0;
        let wall = env;
        if v < wall { (1u32,) } else { (0u32,) }
    }
}

pub struct ReflectHi;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_1> for ReflectHi {
    type Env = f32;
    type Output = f32_1;
    fn apply(env: Wall, inp: f32_1) -> f32_1 {
        let x = inp.0;
        let wall = env;
        // x - (x-wall)*2
        let v = wall * 2. - x;
        (v,)
    }
}

pub struct ReflectLo;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_1> for ReflectLo {
    type Env = f32;
    type Output = f32_1;
    fn apply(env: Wall, inp: f32_1) -> f32_1 {
        let x = inp.0;
        let wall = env;
        // x + (wall-x)*2
        let v = wall * 2. - x;
        (v,)
    }
}

pub struct Negate;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_1> for Negate {
    type Env = ();
    type Output = f32_1;
    fn apply(_env: (), inp: f32_1) -> f32_1 {
        (-inp.0,)
    }
}

pub struct WrapHi;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_1> for WrapHi {
    type Env = Range;
    type Output = f32_1;
    fn apply(env: Range, inp: f32_1) -> f32_1 {
        let x = inp.0;
        let lo = env.lo;
        let hi = env.hi;
        let v = x - (hi - lo);
        (v,)
    }
}

pub struct WrapLo;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_1> for WrapLo {
    type Env = Range;
    type Output = f32_1;
    fn apply(env: Range, inp: f32_1) -> f32_1 {
        let x = inp.0;
        let lo = env.lo;
        let hi = env.hi;
        let v = x + (hi - lo);
        (v,)
    }
}
