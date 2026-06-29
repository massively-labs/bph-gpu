use super::*;

pub struct RungeKutta1<F> {
    _ops: PhantomData<F>,
}

impl<F> RungeKutta1<F> {
    pub const fn new() -> Self {
        Self { _ops: PhantomData }
    }
}

#[cube]
impl<R, F> UnaryOp<R, f32_7> for RungeKutta1<F>
where
    R: Runtime,
    F: UnaryOp<R, f32_7, Output = f32_3>,
{
    // dt, F::Env
    type Env = (f32, F::Env);
    type Output = f32_6;

    fn apply(env: Self::Env, inp: f32_7) -> f32_6 {
        let (dt, env1) = env;
        let (x, y, z, u, v, w, m) = inp;
        let p = (x, y, z);
        let c = (u, v, w);
        let force = F::apply(env1, inp);
        let a = f32_3_div(force, m);
        let new_p = f32_3_add(p, f32_3_mul(c, dt));
        let new_c = f32_3_add(c, f32_3_mul(a, dt));
        let (x, y, z) = new_p;
        let (u, v, w) = new_c;
        (x, y, z, u, v, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runge_kutta_1_constant_force() {
        let before = (0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        let after = <RungeKutta1<massively::op::Constant<f32_3>> as UnaryOp<
            cubecl::wgpu::WgpuRuntime,
            f32_7,
        >>::apply((0.2_f32, (-1.0_f32, 0.0, 0.0)), before);

        assert_eq!(after, (0.2, 0.0, 0.0, 0.8, 0.0, 0.0));
    }
}
