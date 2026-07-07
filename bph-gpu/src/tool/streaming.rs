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
    F: UnaryOp<R, f32_6, Output = f32_3>,
{
    type Output = f32_6;

    fn apply(inp: f32_7) -> f32_6 {
        let (x, y, z, u, v, w, dt) = inp;
        let p = (x, y, z);
        let c = (u, v, w);
        let force = F::apply((x, y, z, u, v, w));
        let new_p = f32_3_add(p, f32_3_mul(c, dt));
        let new_c = f32_3_add(c, f32_3_mul(force, dt));
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
        struct ConstantForce;
        #[cube]
        impl<R: Runtime> UnaryOp<R, f32_6> for ConstantForce {
            type Output = f32_3;
            fn apply(_: f32_6) -> f32_3 {
                (-1.0_f32, 0.0_f32, 0.0_f32)
            }
        }

        let before = (
            0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.2_f32,
        );
        let after =
            <RungeKutta1<ConstantForce> as UnaryOp<cubecl::wgpu::WgpuRuntime, f32_7>>::apply(
                before,
            );

        assert_eq!(
            after,
            (0.2_f32, 0.0_f32, 0.0_f32, 0.8_f32, 0.0_f32, 0.0_f32)
        );
    }
}
