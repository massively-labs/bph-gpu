use bph_gpu::tool::boundary::{
    Negate, OutHi, OutLo, Range, RangeLaunch, ReflectLo, WrapHi, WrapLo,
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
    u0: f32,
    #[clap(long)]
    out: Option<PathBuf>,
}

struct AddVelocity;

#[cube]
impl<R: Runtime> UnaryOp<R, (f32,)> for AddVelocity {
    type Env = f32;
    type Output = (f32,);

    fn apply(u0: f32, inp: (f32,)) -> (f32,) {
        (inp.0 + u0,)
    }
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

struct OutXHiClosed;

#[cube]
impl<R: Runtime> UnaryOp<R, (f32,)> for OutXHiClosed {
    type Env = f32;
    type Output = (u32,);

    fn apply(hi: f32, inp: (f32,)) -> (u32,) {
        if inp.0 >= hi { (1u32,) } else { (0u32,) }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let n = args.n;
    let n_cell = args.m;
    let n_particle = n * n_cell;
    let dt = 1. / n_cell as f32;
    let end_step = (args.fin / dt) as u64;

    let space = Space::new((0., 0., 0.), (dt, 1., 1.), (n_cell, 1, 1));
    let exec = Executor::<cubecl::wgpu::WgpuRuntime>::new(cubecl::wgpu::WgpuDevice::DefaultDevice);

    let mut x = exec.constant(n_particle, 0. as f32)?;
    let mut y = exec.constant(n_particle, 0. as f32)?;
    let mut z = exec.constant(n_particle, 0. as f32)?;
    let mut u = exec.constant(n_particle, 0. as f32)?;
    let mut v = exec.constant(n_particle, 0. as f32)?;
    let mut w = exec.constant(n_particle, 0. as f32)?;
    let mut in_e = exec.constant(n_particle, 0. as f32)?;

    for i in 0..n_cell {
        let cell = space.get_cell_at(i, 0, 0);
        let range = (n * i)..(n * (i + 1));
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
        Zip3(u.slice(..), v.slice(..), w.slice(..)),
        ScaleVelocity,
        3_f32.sqrt(),
        Zip3(u.slice_mut(..), v.slice_mut(..), w.slice_mut(..)),
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

    let mass = exec.constant(x.len() as u32, 1. as f32)?;
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
        Zip1(u.slice(..)),
        AddVelocity,
        args.u0,
        Zip1(u.slice_mut(..)),
    )?;

    for step in 0..end_step {
        let mass = exec.constant(x.len() as u32, 1. as f32)?;
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
        remove_right_outflow(
            &exec, &mut x, &mut y, &mut z, &mut u, &mut v, &mut w, &mut in_e,
        )?;
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
        write_density_1d(out, &counts, n)?;
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
    let Zip1(out_hi) = exec.alloc::<(u32,)>(x.len())?;
    massively::transform(
        exec,
        Zip1(x.slice(..)),
        OutXHiClosed,
        1.,
        Zip1(out_hi.slice_mut(..)),
    )?;
    let tmp = exec.alloc::<(f32, f32, f32, f32, f32, f32, f32)>(x.len())?;
    let n = massively::remove_where(
        exec,
        Zip7(
            x.slice(..),
            y.slice(..),
            z.slice(..),
            u.slice(..),
            v.slice(..),
            w.slice(..),
            in_e.slice(..),
        ),
        out_hi.slice(..),
        tmp.slice_mut(..),
    )?;
    let Zip7(new_x, new_y, new_z, new_u, new_v, new_w, new_in_e) =
        exec.alloc::<(f32, f32, f32, f32, f32, f32, f32)>(n)?;
    let indices = exec.counting(n)?;
    massively::gather(
        exec,
        tmp.slice(..n),
        indices.slice(..),
        Zip7(
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
