use bph_gpu::tool::boundary::{Negate, OutHi, OutLo, Range, ReflectLo, WrapHi, WrapLo};
use bph_gpu::tool::force::NoForce;
use bph_gpu::tool::space::{CalcCellIndex1d, Space};
use bph_gpu::tool::streaming::RungeKutta1;
use clap::Parser;
use cubecl::{Runtime, cube, prelude::*};
use massively::op::{BinaryPredicateOp, UnaryOp};
use massively::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    n: u32,
    m: u32,
    s: f32,
    fin: f32,
    u0: f32,
    #[clap(long)]
    out: Option<PathBuf>,
}

struct AddVelocity;

#[cube]
impl UnaryOp<(f32, f32)> for AddVelocity {
    type Output = f32;

    fn apply(inp: (f32, f32)) -> f32 {
        let (u, u0) = inp;
        u + u0
    }
}

struct ScaleVelocity;

#[cube]
impl UnaryOp<Tuple4<f32, f32, f32, f32>> for ScaleVelocity {
    type Output = Tuple3<f32, f32, f32>;

    fn apply(inp: Tuple4<f32, f32, f32, f32>) -> Self::Output {
        let (u, v, w, scale) = flatten4(inp);
        tuple3(u * scale, v * scale, w * scale)
    }
}

struct OutXHiClosed;

#[cube]
impl UnaryOp<(f32, f32)> for OutXHiClosed {
    type Output = u32;

    fn apply(inp: (f32, f32)) -> u32 {
        let (x, hi) = inp;
        if x >= hi { 1u32 } else { 0u32 }
    }
}

struct LessU32;

#[cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let n = args.n;
    let n_cell = args.m;
    let n_particle = (n * n_cell) as usize;
    let dt = 1. / n_cell as f32;
    let end_step = (args.fin / dt) as u64;

    let space = Space::new((0., 0., 0.), (dt, 1., 1.), (n_cell, 1, 1));
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::DefaultDevice);

    let mut x = exec.full(n_particle, 0. as f32)?;
    let mut y = exec.full(n_particle, 0. as f32)?;
    let mut z = exec.full(n_particle, 0. as f32)?;
    let mut u = exec.full(n_particle, 0. as f32)?;
    let mut v = exec.full(n_particle, 0. as f32)?;
    let mut w = exec.full(n_particle, 0. as f32)?;
    let mut in_e = exec.full(n_particle, 0. as f32)?;

    for i in 0..n_cell {
        let cell = space.get_cell_at(i, 0, 0);
        let range = ((n * i) as usize)..((n * (i + 1)) as usize);
        alloc_position_in_cell(&exec, &mut x, &mut y, &mut z, range, &cell, i);
    }

    bph_gpu::distribution::shell::alloc_shell_rand(
        &exec,
        u.slice_mut(..),
        v.slice_mut(..),
        w.slice_mut(..),
        2,
    );
    massively::transform(
        &exec,
        zip4(
            u.slice(..),
            v.slice(..),
            w.slice(..),
            massively::lazy::constant(3_f32.sqrt()).take(u.len() as u32),
        ),
        ScaleVelocity,
        zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )?;

    let idx = calc_idx(&exec, &x, &y, &z, dt, n_cell)?;
    let sorted_idx = exec.alloc::<u32>(idx.len());
    let (sorted_x, sorted_y, sorted_z, sorted_u, sorted_v, sorted_w, _sorted_in_e) =
        alloc_f32x7(&exec, idx.len());
    massively::sort_by_key(
        &exec,
        idx.slice(..),
        zip7(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            u.slice(..),
            v.slice(..),
            w.slice(..),
            in_e.slice(..),
        ),
        LessU32,
        sorted_idx.slice_mut(..),
        zip7(
            sorted_x.slice_mut(..),
            sorted_y.slice_mut(..),
            sorted_z.slice_mut(..),
            sorted_u.slice_mut(..),
            sorted_v.slice_mut(..),
            sorted_w.slice_mut(..),
            _sorted_in_e.slice_mut(..),
        ),
    )?;
    x = sorted_x;
    y = sorted_y;
    z = sorted_z;
    u = sorted_u;
    v = sorted_v;
    w = sorted_w;

    let mass = exec.full(x.len(), 1. as f32)?;
    in_e = bph_gpu::tool::calc_in_e::calc_in_e(
        &exec,
        u.slice(..),
        v.slice(..),
        w.slice(..),
        mass.slice(..),
        sorted_idx.slice(..),
        n_cell,
        args.s,
    );
    massively::transform(
        &exec,
        zip2(
            u.slice(..),
            massively::lazy::constant(args.u0).take(u.len() as u32),
        ),
        AddVelocity,
        u.slice_mut(..),
    )?;

    for step in 0..end_step {
        let mass = exec.full(x.len(), 1. as f32)?;
        let idx = calc_idx(&exec, &x, &y, &z, dt, n_cell)?;
        let sorted_idx = exec.alloc::<u32>(idx.len());
        let (sorted_x, sorted_y, sorted_z, sorted_u, sorted_v, sorted_w, sorted_in_e) =
            alloc_f32x7(&exec, idx.len());
        massively::sort_by_key(
            &exec,
            idx.slice(..),
            zip7(
                x.slice(..),
                y.slice(..),
                z.slice(..),
                u.slice(..),
                v.slice(..),
                w.slice(..),
                in_e.slice(..),
            ),
            LessU32,
            sorted_idx.slice_mut(..),
            zip7(
                sorted_x.slice_mut(..),
                sorted_y.slice_mut(..),
                sorted_z.slice_mut(..),
                sorted_u.slice_mut(..),
                sorted_v.slice_mut(..),
                sorted_w.slice_mut(..),
                sorted_in_e.slice_mut(..),
            ),
        )?;

        x = sorted_x;
        y = sorted_y;
        z = sorted_z;
        u = sorted_u;
        v = sorted_v;
        w = sorted_w;
        in_e = sorted_in_e;

        bph_gpu::bph(
            &exec,
            u.slice_mut(..),
            v.slice_mut(..),
            w.slice_mut(..),
            mass.slice(..),
            in_e.slice_mut(..),
            sorted_idx.slice(..),
            n_cell,
            args.s,
            step,
        );

        massively::transform(
            &exec,
            zip7(
                x.slice(..),
                y.slice(..),
                z.slice(..),
                u.slice(..),
                v.slice(..),
                w.slice(..),
                massively::lazy::constant(dt).take(x.len() as u32),
            ),
            RungeKutta1::<NoForce>::new(),
            zip6(
                x.slice_mut(..),
                y.slice_mut(..),
                z.slice_mut(..),
                u.slice_mut(..),
                v.slice_mut(..),
                w.slice_mut(..),
            ),
        )?;

        apply_periodic(&exec, &mut y, 0., 1.)?;
        apply_periodic(&exec, &mut z, 0., 1.)?;
        apply_reflect_lo_x(&exec, &mut x, &mut u)?;
        remove_right_outflow(
            &exec, &mut x, &mut y, &mut z, &mut u, &mut v, &mut w, &mut in_e,
        )?;
    }

    if let Some(out) = args.out {
        let idx = calc_idx(&exec, &x, &y, &z, dt, n_cell)?;
        let sorted_idx = exec.alloc::<u32>(idx.len());
        massively::sort(&exec, idx.slice(..), LessU32, sorted_idx.slice_mut(..))?;
        let counts = exec.full(n_cell as usize, 0_u32)?;
        bph_gpu::algorithm::bucket_counting(&exec, sorted_idx.slice(..), counts.slice_mut(..))?;
        let counts = exec.to_host(&counts)?;
        write_density_1d(out, &counts, n)?;
    }

    Ok(())
}

fn alloc_position_in_cell<R: Runtime>(
    exec: &Executor<R>,
    x: &mut DeviceVec<R, f32>,
    y: &mut DeviceVec<R, f32>,
    z: &mut DeviceVec<R, f32>,
    range: std::ops::Range<usize>,
    cell: &bph_gpu::tool::space::Cell,
    seed: u32,
) {
    bph_gpu::tool::random::alloc_uniform_random(
        exec,
        x.slice_mut(range.clone()),
        Range {
            lo: cell.x_min(),
            hi: cell.x_max(),
        },
        seed as u64,
    );
    bph_gpu::tool::random::alloc_uniform_random(
        exec,
        y.slice_mut(range.clone()),
        Range {
            lo: cell.y_min(),
            hi: cell.y_max(),
        },
        (seed + 1) as u64,
    );
    bph_gpu::tool::random::alloc_uniform_random(
        exec,
        z.slice_mut(range),
        Range {
            lo: cell.z_min(),
            hi: cell.z_max(),
        },
        (seed + 2) as u64,
    );
}

fn calc_idx<R: Runtime>(
    exec: &Executor<R>,
    x: &DeviceVec<R, f32>,
    _y: &DeviceVec<R, f32>,
    _z: &DeviceVec<R, f32>,
    dt: f32,
    n_cell: u32,
) -> bph_gpu::Error<DeviceVec<R, u32>> {
    let idx = exec.alloc::<u32>(x.len());
    massively::transform(
        exec,
        zip4(
            x.slice(..),
            massively::lazy::constant(0.0_f32).take(x.len() as u32),
            massively::lazy::constant(dt).take(x.len() as u32),
            massively::lazy::constant(n_cell).take(x.len() as u32),
        ),
        CalcCellIndex1d,
        idx.slice_mut(..),
    )?;
    Ok(idx)
}

type F32x7<R> = (
    DeviceVec<R, f32>,
    DeviceVec<R, f32>,
    DeviceVec<R, f32>,
    DeviceVec<R, f32>,
    DeviceVec<R, f32>,
    DeviceVec<R, f32>,
    DeviceVec<R, f32>,
);

fn alloc_f32x7<R: Runtime>(exec: &Executor<R>, len: usize) -> F32x7<R> {
    (
        exec.alloc::<f32>(len),
        exec.alloc::<f32>(len),
        exec.alloc::<f32>(len),
        exec.alloc::<f32>(len),
        exec.alloc::<f32>(len),
        exec.alloc::<f32>(len),
        exec.alloc::<f32>(len),
    )
}

fn apply_periodic<R: Runtime>(
    exec: &Executor<R>,
    values: &mut DeviceVec<R, f32>,
    lo: f32,
    hi: f32,
) -> bph_gpu::Error<()> {
    let out_lo = massively::lazy::transform(
        zip2(
            values.slice(..),
            massively::lazy::constant(lo).take(values.len() as u32),
        ),
        OutLo,
    );
    massively::transform_where(
        exec,
        zip3(
            values.slice(..),
            massively::lazy::constant(lo).take(values.len() as u32),
            massively::lazy::constant(hi).take(values.len() as u32),
        ),
        WrapLo,
        out_lo,
        values.slice_mut(..),
    )?;

    let out_hi = massively::lazy::transform(
        zip2(
            values.slice(..),
            massively::lazy::constant(hi).take(values.len() as u32),
        ),
        OutHi,
    );
    massively::transform_where(
        exec,
        zip3(
            values.slice(..),
            massively::lazy::constant(lo).take(values.len() as u32),
            massively::lazy::constant(hi).take(values.len() as u32),
        ),
        WrapHi,
        out_hi,
        values.slice_mut(..),
    )?;

    Ok(())
}

fn apply_reflect_lo_x<R: Runtime>(
    exec: &Executor<R>,
    x: &mut DeviceVec<R, f32>,
    u: &mut DeviceVec<R, f32>,
) -> bph_gpu::Error<()> {
    let out_lo = massively::lazy::transform(
        zip2(
            x.slice(..),
            massively::lazy::constant(0.0_f32).take(x.len() as u32),
        ),
        OutLo,
    );
    massively::transform_where(exec, u.slice(..), Negate, out_lo, u.slice_mut(..))?;
    let out_lo = massively::lazy::transform(
        zip2(
            x.slice(..),
            massively::lazy::constant(0.0_f32).take(x.len() as u32),
        ),
        OutLo,
    );
    massively::transform_where(
        exec,
        zip2(
            x.slice(..),
            massively::lazy::constant(0.0_f32).take(x.len() as u32),
        ),
        ReflectLo,
        out_lo,
        x.slice_mut(..),
    )?;

    Ok(())
}

fn remove_right_outflow<R: Runtime>(
    exec: &Executor<R>,
    x: &mut DeviceVec<R, f32>,
    y: &mut DeviceVec<R, f32>,
    z: &mut DeviceVec<R, f32>,
    u: &mut DeviceVec<R, f32>,
    v: &mut DeviceVec<R, f32>,
    w: &mut DeviceVec<R, f32>,
    in_e: &mut DeviceVec<R, f32>,
) -> bph_gpu::Error<()> {
    let out_hi = massively::lazy::transform(
        zip2(
            x.slice(..),
            massively::lazy::constant(1.0_f32).take(x.len() as u32),
        ),
        OutXHiClosed,
    );
    let (tmp_x, tmp_y, tmp_z, tmp_u, tmp_v, tmp_w, tmp_in_e) = alloc_f32x7(exec, x.len());
    let n = massively::remove_where(
        exec,
        zip7(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            u.slice(..),
            v.slice(..),
            w.slice(..),
            in_e.slice(..),
        ),
        out_hi,
        zip7(
            tmp_x.slice_mut(..),
            tmp_y.slice_mut(..),
            tmp_z.slice_mut(..),
            tmp_u.slice_mut(..),
            tmp_v.slice_mut(..),
            tmp_w.slice_mut(..),
            tmp_in_e.slice_mut(..),
        ),
    )?;
    let (new_x, new_y, new_z, new_u, new_v, new_w, new_in_e) = alloc_f32x7(exec, n as usize);
    massively::copy_where(
        exec,
        zip7(
            tmp_x.slice(..n as usize),
            tmp_y.slice(..n as usize),
            tmp_z.slice(..n as usize),
            tmp_u.slice(..n as usize),
            tmp_v.slice(..n as usize),
            tmp_w.slice(..n as usize),
            tmp_in_e.slice(..n as usize),
        ),
        massively::lazy::constant(1u32).take(n),
        zip7(
            new_x.slice_mut(..),
            new_y.slice_mut(..),
            new_z.slice_mut(..),
            new_u.slice_mut(..),
            new_v.slice_mut(..),
            new_w.slice_mut(..),
            new_in_e.slice_mut(..),
        ),
    )?;

    *x = new_x;
    *y = new_y;
    *z = new_z;
    *u = new_u;
    *v = new_v;
    *w = new_w;
    *in_e = new_in_e;

    Ok(())
}

fn write_density_1d(
    out: PathBuf,
    counts: &[u32],
    particles_per_unit_density: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(out)?;
    let mut writer = BufWriter::new(file);
    for count in counts {
        writeln!(
            writer,
            "{}",
            *count as f32 / particles_per_unit_density as f32
        )?;
    }
    Ok(())
}
