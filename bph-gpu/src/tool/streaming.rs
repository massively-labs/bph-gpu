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
impl<F> UnaryOp<f32_7> for RungeKutta1<F>
where
    F: UnaryOp<f32_6, Output = f32_3>,
{
    type Output = f32_6;

    fn apply(inp: f32_7) -> f32_6 {
        let (x, y, z, u, v, w, dt) = flatten7(inp);
        let p = tuple3(x, y, z);
        let c = tuple3(u, v, w);
        let force = F::apply(tuple6(x, y, z, u, v, w));
        let new_p = f32_3_add(p, f32_3_mul(c, dt));
        let new_c = f32_3_add(c, f32_3_mul(force, dt));
        let (x, y, z) = flatten3(new_p);
        let (u, v, w) = flatten3(new_c);
        tuple6(x, y, z, u, v, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runge_kutta_1_constant_force() {
        struct ConstantForce;
        #[cube]
        impl UnaryOp<f32_6> for ConstantForce {
            type Output = f32_3;
            fn apply(_: f32_6) -> f32_3 {
                tuple3(-1.0_f32, 0.0_f32, 0.0_f32)
            }
        }

        let before = tuple7(
            0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 0.2_f32,
        );
        let after = <RungeKutta1<ConstantForce> as UnaryOp<f32_7>>::apply(before);

        assert_eq!(
            flatten6(after),
            (0.2_f32, 0.0_f32, 0.0_f32, 0.8_f32, 0.0_f32, 0.0_f32)
        );
    }
}
