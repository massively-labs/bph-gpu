use bph_gpu::tool::boundary::{
    Negate, OutHi, OutLo, Range, RangeLaunch, ReflectHi, ReflectLo, WrapHi, WrapLo,
};
use bph_gpu::tool::force::NoForce;
use bph_gpu::tool::space::{CalcCellIndex1d, Space, SpaceLaunch};
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

struct ScaleVelocity;

#[cube]
impl<R: Runtime> UnaryOp<R, (f32, f32, f32)> for ScaleVelocity {
    type Env = f32;
    type Output = (f32, f32, f32);

    fn apply(scale: f32, inp: (f32, f32, f32)) -> (f32, f32, f32) {
        (inp.0 * scale, inp.1 * scale, inp.2 * scale)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let n = args.n;
    let m = args.m;
    let n_particle = (1 + 8) * n * m;
    let n_cell = 2 * m;
    let dt = 1. / n_cell as f32;
    let end_step = (args.fin / dt) as u64;
    let sep = 8 * n * m;

    let space = Space::new((0., 0., 0.), (dt, 1., 1.), (n_cell, 1, 1));
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::DefaultDevice);

    let mass = exec.constant(n_particle, 1. as f32)?;
    let mut x = exec.constant(n_particle, 0. as f32)?;
    let mut y = exec.constant(n_particle, 0. as f32)?;
    let mut z = exec.constant(n_particle, 0. as f32)?;
    let mut u = exec.constant(n_particle, 0. as f32)?;
    let mut v = exec.constant(n_particle, 0. as f32)?;
    let mut w = exec.constant(n_particle, 0. as f32)?;
    let mut in_e = exec.constant(n_particle, 0. as f32)?;

    for i in 0..m {
        let cell = space.get_cell_at(i, 0, 0);
        let range = (8 * n * i)..(8 * n * (i + 1));
        alloc_position_in_cell(&exec, &mut x, &mut y, &mut z, range, &cell, i);
    }

    for i in 0..m {
        let cell = space.get_cell_at(m + i, 0, 0);
        let range = (sep + n * i)..(sep + n * (i + 1));
        alloc_position_in_cell(&exec, &mut x, &mut y, &mut z, range, &cell, m + i);
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
        Zip3(u.slice(0..sep), v.slice(0..sep), w.slice(0..sep)),
        ScaleVelocity,
        3_f32.sqrt(),
        Zip3(
            u.slice_mut(0..sep),
            v.slice_mut(0..sep),
            w.slice_mut(0..sep),
        ),
    )?;
    massively::transform(
        &exec,
        Zip3(
            u.slice(sep..n_particle),
            v.slice(sep..n_particle),
            w.slice(sep..n_particle),
        ),
        ScaleVelocity,
        3_f32.sqrt() / 1.25_f32.sqrt(),
        Zip3(
            u.slice_mut(sep..n_particle),
            v.slice_mut(sep..n_particle),
            w.slice_mut(sep..n_particle),
        ),
    )?;

    let Zip1(idx) = calc_idx(&exec, &x, &y, &z, dt, n_cell)?;
    let Zip1(sorted_idx) = exec.alloc::<(u32,)>(idx.len())?;
    let Zip7(sorted_x, sorted_y, sorted_z, sorted_u, sorted_v, sorted_w, _sorted_in_e) =
        exec.alloc::<(f32, f32, f32, f32, f32, f32, f32)>(idx.len())?;
    massively::sort_by_key(
        &exec,
        Zip1(idx.slice(..)),
        Zip7(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            u.slice(..),
            v.slice(..),
            w.slice(..),
            in_e.slice(..),
        ),
        massively::op::Less,
        Zip1(sorted_idx.slice_mut(..)),
        Zip7(
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

    for step in 0..end_step {
        let Zip1(idx) = calc_idx(&exec, &x, &y, &z, dt, n_cell)?;
        let Zip1(sorted_idx) = exec.alloc::<(u32,)>(idx.len())?;
        let Zip7(sorted_x, sorted_y, sorted_z, sorted_u, sorted_v, sorted_w, sorted_in_e) =
            exec.alloc::<(f32, f32, f32, f32, f32, f32, f32)>(idx.len())?;
        massively::sort_by_key(
            &exec,
            Zip1(idx.slice(..)),
            Zip7(
                x.slice(..),
                y.slice(..),
                z.slice(..),
                u.slice(..),
                v.slice(..),
                w.slice(..),
                in_e.slice(..),
            ),
            massively::op::Less,
            Zip1(sorted_idx.slice_mut(..)),
            Zip7(
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
            Zip7(
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
            Zip6(
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
        apply_reflect_hi_x(&exec, &mut x, &mut u)?;
    }

    if let Some(out) = args.out {
        let Zip1(idx) = calc_idx(&exec, &x, &y, &z, dt, n_cell)?;
        let Zip1(sorted_idx) = exec.alloc::<(u32,)>(idx.len())?;
        massively::sort(
            &exec,
            Zip1(idx.slice(..)),
            massively::op::Less,
            Zip1(sorted_idx.slice_mut(..)),
        )?;
        let counts = exec.constant(n_cell, 0_u32)?;
        bph_gpu::algorithm::bucket_counting(&exec, sorted_idx.slice(..), counts.slice_mut(..))?;
        let counts = exec.to_host(&counts)?;
        write_density_1d(out, &counts, 8 * n)?;
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
    dt: f32,
    n_cell: u32,
) -> bph_gpu::Error<Zip1<DeviceVec<R, u32>>> {
    let Zip1(idx) = exec.alloc::<(u32,)>(x.len())?;
    massively::transform(
        exec,
        Zip3(x.slice(..), y.slice(..), z.slice(..)),
        CalcCellIndex1d,
        SpaceLaunch::new((0., 0., 0.), (dt, 1., 1.), (n_cell, 1, 1)),
        Zip1(idx.slice_mut(..)),
    )?;
    Ok(Zip1(idx))
}

fn apply_periodic<R: Runtime>(
    exec: &Executor<R>,
    values: &mut DeviceVec<R, f32>,
    lo: f32,
    hi: f32,
) -> bph_gpu::Error<()> {
    let Zip1(out_lo) = exec.alloc::<(u32,)>(values.len())?;
    massively::transform(
        exec,
        Zip1(values.slice(..)),
        OutLo,
        lo,
        Zip1(out_lo.slice_mut(..)),
    )?;
    massively::transform_where(
        exec,
        Zip1(values.slice(..)),
        WrapLo,
        RangeLaunch::new(lo, hi),
        out_lo.slice(..),
        Zip1(values.slice_mut(..)),
    )?;

    let Zip1(out_hi) = exec.alloc::<(u32,)>(values.len())?;
    massively::transform(
        exec,
        Zip1(values.slice(..)),
        OutHi,
        hi,
        Zip1(out_hi.slice_mut(..)),
    )?;
    massively::transform_where(
        exec,
        Zip1(values.slice(..)),
        WrapHi,
        RangeLaunch::new(lo, hi),
        out_hi.slice(..),
        Zip1(values.slice_mut(..)),
    )?;

    Ok(())
}

fn apply_reflect_lo_x<R: Runtime>(
    exec: &Executor<R>,
    x: &mut DeviceVec<R, f32>,
    u: &mut DeviceVec<R, f32>,
) -> bph_gpu::Error<()> {
    let Zip1(out_lo) = exec.alloc::<(u32,)>(x.len())?;
    massively::transform(
        exec,
        Zip1(x.slice(..)),
        OutLo,
        0.,
        Zip1(out_lo.slice_mut(..)),
    )?;
    massively::transform_where(
        exec,
        Zip1(u.slice(..)),
        Negate,
        (),
        out_lo.slice(..),
        Zip1(u.slice_mut(..)),
    )?;
    massively::transform_where(
        exec,
        Zip1(x.slice(..)),
        ReflectLo,
        0.,
        out_lo.slice(..),
        Zip1(x.slice_mut(..)),
    )?;

    Ok(())
}

fn apply_reflect_hi_x<R: Runtime>(
    exec: &Executor<R>,
    x: &mut DeviceVec<R, f32>,
    u: &mut DeviceVec<R, f32>,
) -> bph_gpu::Error<()> {
    let Zip1(out_hi) = exec.alloc::<(u32,)>(x.len())?;
    massively::transform(
        exec,
        Zip1(x.slice(..)),
        OutHi,
        1.,
        Zip1(out_hi.slice_mut(..)),
    )?;
    massively::transform_where(
        exec,
        Zip1(u.slice(..)),
        Negate,
        (),
        out_hi.slice(..),
        Zip1(u.slice_mut(..)),
    )?;
    massively::transform_where(
        exec,
        Zip1(x.slice(..)),
        ReflectHi,
        1.,
        out_hi.slice(..),
        Zip1(x.slice_mut(..)),
    )?;

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
