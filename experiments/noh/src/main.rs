use bph_gpu::tool::boundary::Range;
use bph_gpu::tool::force::NoForce;
use bph_gpu::tool::space::Space;
use bph_gpu::tool::streaming::RungeKutta1;
use clap::Parser;
use cubecl::{Runtime, cube, prelude::*};
use massively::op::UnaryOp;
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

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone))]
struct NohSpace {
    origin: (f32, f32, f32),
    space: (f32, f32, f32),
    dim: (u32, u32, u32),
}

struct CalcCellIndexNoh2d;

#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32)> for CalcCellIndexNoh2d {
    type Env = NohSpace;
    type Output = (u32,);

    fn apply(env: NohSpace, p: (f32, f32, f32)) -> (u32,) {
        let i = ((p.0 - env.origin.0) / env.space.0) as u32;
        let j = ((p.1 - env.origin.1) / env.space.1) as u32;
        let k = ((p.2 - env.origin.2) / env.space.2) as u32;
        (i * env.dim.1 * env.dim.2 + j * env.dim.2 + k,)
    }
}

struct OutOfCircle;

#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32)> for OutOfCircle {
    type Env = (f32, f32, f32, f32);
    type Output = (u32,);

    fn apply(env: (f32, f32, f32, f32), p: (f32, f32, f32)) -> (u32,) {
        let dx = p.0 - env.0;
        let dy = p.1 - env.1;
        let dz = p.2 - env.2;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        if distance > env.3 { (1u32,) } else { (0u32,) }
    }
}

struct VelocityTowardCenter;

#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32)> for VelocityTowardCenter {
    type Env = (f32, f32, f32);
    type Output = (f32, f32, f32);

    fn apply(center: (f32, f32, f32), p: (f32, f32, f32)) -> (f32, f32, f32) {
        let dx = center.0 - p.0;
        let dy = center.1 - p.1;
        let dz = center.2 - p.2;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len == 0.0_f32 {
            (0.0_f32, 0.0_f32, 0.0_f32)
        } else {
            (dx / len, dy / len, dz / len)
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let n = args.n;
    let m = args.m;
    let rad = 1.;
    let width = 2 * m;
    let n_cell = width * width;
    let n_particle = n * n_cell;
    let cell_size = rad / m as f32;
    let dt = 1. / m as f32;
    let end_step = (args.fin / dt) as u64;
    let center = (rad, rad, 0.5);

    let space = Space::new((0., 0., 0.), (cell_size, cell_size, 1.), (width, width, 1));
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::DefaultDevice);

    let mut x = exec.constant(n_particle, 0. as f32)?;
    let mut y = exec.constant(n_particle, 0. as f32)?;
    let mut z = exec.constant(n_particle, 0. as f32)?;
    let mut u = exec.constant(n_particle, 0. as f32)?;
    let mut v = exec.constant(n_particle, 0. as f32)?;
    let mut w = exec.constant(n_particle, 0. as f32)?;
    let mut in_e = exec.constant(n_particle, 0. as f32)?;

    for i in 0..width {
        for j in 0..width {
            let cell = space.get_cell_at(i, j, 0);
            let cell_index = i * width + j;
            let range = (n * cell_index)..(n * (cell_index + 1));
            alloc_position_in_cell(&exec, &mut x, &mut y, &mut z, range, &cell, i);
        }
    }

    remove_out_of_circle(
        &exec, &mut x, &mut y, &mut z, &mut u, &mut v, &mut w, &mut in_e, center, rad,
    )?;

    massively::transform(
        &exec,
        SoA3(x.slice(..), y.slice(..), z.slice(..)),
        VelocityTowardCenter,
        center,
        SoA3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
    )?;

    for step in 0..end_step {
        let mass = exec.constant(x.len() as u32, 1. as f32)?;
        let SoA1(idx) = calc_idx(&exec, &x, &y, &z, cell_size, width)?;
        let (
            SoA1(sorted_idx),
            SoA7(sorted_x, sorted_y, sorted_z, sorted_u, sorted_v, sorted_w, sorted_in_e),
        ) = massively::sort_by_key(
            &exec,
            SoA1(idx.slice(..)),
            SoA7(
                x.slice(..),
                y.slice(..),
                z.slice(..),
                u.slice(..),
                v.slice(..),
                w.slice(..),
                in_e.slice(..),
            ),
            massively::op::Less,
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
            SoA7(
                x.slice(..),
                y.slice(..),
                z.slice(..),
                u.slice(..),
                v.slice(..),
                w.slice(..),
                mass.slice(..),
            ),
            RungeKutta1::<NoForce>::new(),
            (dt, ()),
            SoA6(
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
        let SoA1(idx) = calc_idx(&exec, &x, &y, &z, cell_size, width)?;
        let SoA1(sorted_idx) = massively::sort(&exec, SoA1(idx.slice(..)), massively::op::Less)?;
        let counts = bph_gpu::algorithm::bucket_counting(&exec, sorted_idx.slice(..), n_cell);
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
    range: std::ops::Range<u32>,
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
) -> bph_gpu::Error<SoA1<DeviceVec<R, u32>>> {
    massively::map(
        exec,
        SoA3(x.slice(..), y.slice(..), z.slice(..)),
        CalcCellIndexNoh2d,
        NohSpaceLaunch::new((0., 0., 0.), (cell_size, cell_size, 1.), (width, width, 1)),
    )
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
    let SoA1(out_of_circle) = massively::map(
        exec,
        SoA3(x.slice(..), y.slice(..), z.slice(..)),
        OutOfCircle,
        (center.0, center.1, center.2, rad),
    )?;
    let SoA7(new_x, new_y, new_z, new_u, new_v, new_w, new_in_e) = massively::remove_where(
        exec,
        SoA7(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            u.slice(..),
            v.slice(..),
            w.slice(..),
            in_e.slice(..),
        ),
        out_of_circle.slice(..),
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
