use bph_gpu::tool::boundary::Range;
use bph_gpu::tool::force::NoForce;
use bph_gpu::tool::space::Space;
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
    #[clap(long)]
    out: Option<PathBuf>,
}

struct CalcCellIndexNoh2d;

#[cube]
impl UnaryOp<Tuple5<f32, f32, f32, f32, u32>> for CalcCellIndexNoh2d {
    type Output = u32;

    fn apply(input: Tuple5<f32, f32, f32, f32, u32>) -> u32 {
        let (x, y, _z, cell_size, width) = flatten5(input);
        let i = (x / cell_size) as u32;
        let j = (y / cell_size) as u32;
        i * width + j
    }
}

struct OutOfCircle;

#[cube]
impl UnaryOp<Tuple7<f32, f32, f32, f32, f32, f32, f32>> for OutOfCircle {
    type Output = bool;

    fn apply(input: Tuple7<f32, f32, f32, f32, f32, f32, f32>) -> bool {
        let (x, y, z, cx, cy, cz, rad) = flatten7(input);
        let dx = x - cx;
        let dy = y - cy;
        let dz = z - cz;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        distance > rad
    }
}

struct VelocityTowardCenter;

#[cube]
impl UnaryOp<Tuple6<f32, f32, f32, f32, f32, f32>> for VelocityTowardCenter {
    type Output = Tuple3<f32, f32, f32>;

    fn apply(input: Tuple6<f32, f32, f32, f32, f32, f32>) -> Self::Output {
        let (x, y, z, cx, cy, cz) = flatten6(input);
        let dx = cx - x;
        let dy = cy - y;
        let dz = cz - z;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len == 0.0_f32 {
            tuple3(0.0_f32, 0.0_f32, 0.0_f32)
        } else {
            tuple3(dx / len, dy / len, dz / len)
        }
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
    let m = args.m;
    let rad = 1.;
    let width = 2 * m;
    let n_cell = width * width;
    let n_particle = (n * n_cell) as usize;
    let cell_size = rad / m as f32;
    let dt = 1. / m as f32;
    let end_step = (args.fin / dt) as u64;
    let center = (rad, rad, 0.5);

    let space = Space::new((0., 0., 0.), (cell_size, cell_size, 1.), (width, width, 1));
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::DefaultDevice);

    let mut x = exec.full(n_particle, 0. as f32)?;
    let mut y = exec.full(n_particle, 0. as f32)?;
    let mut z = exec.full(n_particle, 0. as f32)?;
    let mut u = exec.full(n_particle, 0. as f32)?;
    let mut v = exec.full(n_particle, 0. as f32)?;
    let mut w = exec.full(n_particle, 0. as f32)?;
    let mut in_e = exec.full(n_particle, 0. as f32)?;

    for i in 0..width {
        for j in 0..width {
            let cell = space.get_cell_at(i, j, 0);
            let cell_index = i * width + j;
            let range = ((n * cell_index) as usize)..((n * (cell_index + 1)) as usize);
            alloc_position_in_cell(&exec, &mut x, &mut y, &mut z, range, &cell, i);
        }
    }

    remove_out_of_circle(
        &exec, &mut x, &mut y, &mut z, &mut u, &mut v, &mut w, &mut in_e, center, rad,
    )?;

    bph_gpu::algorithm::transform_into(
        &exec,
        zip6(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            massively::lazy::constant(center.0).take(x.len()),
            massively::lazy::constant(center.1).take(x.len()),
            massively::lazy::constant(center.2).take(x.len()),
        ),
        VelocityTowardCenter,
        zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )?;

    for step in 0..end_step {
        let mass = exec.full(x.len(), 1. as f32)?;
        let idx = calc_idx(&exec, &x, &y, &z, cell_size, width)?;
        let (sorted_idx, sorted_values) = bph_gpu::algorithm::sort_by_key_with_keys(
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
        )?;
        let (sorted_x, sorted_y, sorted_z, sorted_u, sorted_v, sorted_w, sorted_in_e) =
            unzip7(sorted_values);

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

        bph_gpu::algorithm::transform_into(
            &exec,
            zip7(
                x.slice(..),
                y.slice(..),
                z.slice(..),
                u.slice(..),
                v.slice(..),
                w.slice(..),
                massively::lazy::constant(dt).take(x.len()),
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
    }

    if let Some(out) = args.out {
        let idx = calc_idx(&exec, &x, &y, &z, cell_size, width)?;
        let sorted_idx = massively::vector::sort(&exec, idx.slice(..), LessU32)?;
        let counts = exec.full(n_cell as usize, 0_u32)?;
        bph_gpu::algorithm::bucket_counting(&exec, sorted_idx.slice(..), counts.slice_mut(..))?;
        let counts = exec.to_host(&counts)?;
        write_density_matrix(out, &counts, width, n)?;
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
    y: &DeviceVec<R, f32>,
    z: &DeviceVec<R, f32>,
    cell_size: f32,
    width: u32,
) -> bph_gpu::Error<DeviceVec<R, u32>> {
    let idx = exec.alloc::<u32>(x.len());
    bph_gpu::algorithm::transform_into(
        exec,
        zip5(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            massively::lazy::constant(cell_size).take(x.len()),
            massively::lazy::constant(width).take(x.len()),
        ),
        CalcCellIndexNoh2d,
        idx.slice_mut(..),
    )?;
    Ok(idx)
}

fn remove_out_of_circle<R: Runtime>(
    exec: &Executor<R>,
    x: &mut DeviceVec<R, f32>,
    y: &mut DeviceVec<R, f32>,
    z: &mut DeviceVec<R, f32>,
    u: &mut DeviceVec<R, f32>,
    v: &mut DeviceVec<R, f32>,
    w: &mut DeviceVec<R, f32>,
    in_e: &mut DeviceVec<R, f32>,
    center: (f32, f32, f32),
    rad: f32,
) -> bph_gpu::Error<()> {
    let out_of_circle = massively::lazy::transform(
        zip7(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            massively::lazy::constant(center.0).take(x.len()),
            massively::lazy::constant(center.1).take(x.len()),
            massively::lazy::constant(center.2).take(x.len()),
            massively::lazy::constant(rad).take(x.len()),
        ),
        OutOfCircle,
    );
    let filtered = massively::vector::remove_where(
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
        out_of_circle,
    )?;
    let (new_x, new_y, new_z, new_u, new_v, new_w, new_in_e) = unzip7(filtered);

    *x = new_x;
    *y = new_y;
    *z = new_z;
    *u = new_u;
    *v = new_v;
    *w = new_w;
    *in_e = new_in_e;

    Ok(())
}

fn write_density_matrix(
    out: PathBuf,
    counts: &[u32],
    width: u32,
    n: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(out)?;
    let mut writer = BufWriter::new(file);
    for i in 0..width {
        for j in 0..width {
            let index = (i * width + j) as usize;
            if j > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{}", counts[index] as f32 / n as f32)?;
        }
        writeln!(writer)?;
    }
    Ok(())
}
