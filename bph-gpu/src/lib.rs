use cubecl::{Runtime, cube, prelude::*};
use massively::{
    op::{BinaryPredicateOp, ReductionOp, UnaryOp},
    prelude::*,
};
use std::marker::PhantomData;

macro_rules! tuple {
    ($ty:ty, 1) => {
        ($ty,)
    };
    ($ty:ty, 2) => {
        ($ty, $ty)
    };
    ($ty:ty, 3) => {
        ($ty, $ty, $ty)
    };
    ($ty:ty, 4) => {
        ($ty, $ty, $ty, $ty)
    };
    ($ty:ty, 5) => {
        ($ty, $ty, $ty, $ty, $ty)
    };
    ($ty:ty, 6) => {
        ($ty, $ty, $ty, $ty, $ty, $ty)
    };
    ($ty:ty, 7) => {
        ($ty, $ty, $ty, $ty, $ty, $ty, $ty)
    };
}

type f32_1 = tuple!(f32, 1);
type f32_2 = tuple!(f32, 2);
type f32_3 = tuple!(f32, 3);
type f32_4 = tuple!(f32, 4);
type f32_5 = tuple!(f32, 5);
type f32_6 = tuple!(f32, 6);
type f32_7 = tuple!(f32, 7);

#[cube]
fn f32_3_add(lhs: f32_3, rhs: f32_3) -> f32_3 {
    (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
}

#[cube]
fn f32_3_sub(lhs: f32_3, rhs: f32_3) -> f32_3 {
    (lhs.0 - rhs.0, lhs.1 - rhs.1, lhs.2 - rhs.2)
}

#[cube]
fn f32_3_div(lhs: f32_3, rhs: f32) -> f32_3 {
    (lhs.0 / rhs, lhs.1 / rhs, lhs.2 / rhs)
}

#[cube]
fn f32_3_mul(lhs: f32_3, rhs: f32) -> f32_3 {
    (lhs.0 * rhs, lhs.1 * rhs, lhs.2 * rhs)
}

#[cube]
fn f32_safe_div(a: f32, b: f32) -> f32 {
    if b == 0. { 0. as f32 } else { a / b }
}

pub mod algorithm;
mod bph;
mod calc_kin_e;
mod calc_total_e;
mod common;
pub mod distribution;
mod relax;
pub mod tool;
mod velocity;

pub type Error<T> = std::result::Result<T, massively::Error>;

pub use bph::bph;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn test_executor() -> Executor<cubecl::wgpu::WgpuRuntime> {
    Executor::new(cubecl::wgpu::WgpuDevice::Cpu)
}
