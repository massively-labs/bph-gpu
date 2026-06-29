# BPH

Boltzmann Particle Hydrodynamics (BPH) on GPU.

## What is BPH?

BPH stands for Boltzmann Particle Hydrodynamics.
It's a very fast algorithm for fluid simulation 
and its applications are in the space.

There are presentations to learn about BPH.

- [Boltzmann Particle Hydrodynamics (Dr. Matsuda)](https://www.cps-jp.org/modules/mosir/player.php?v=20111027_matsuda)
- [An Engineering Application of BPH Method (Dr. Isaka)](https://www.cps-jp.org/modules/mosir/player.php?v=20111027_isaka).

Also see bph.pdf slides.

## Architecture

This library is a rewrite of [bphcuda](https://github.com/akiradeveloper/bphcuda) (my master's work) in Rust. Before starting this project, I built [massively](https://github.com/akiradeveloper/massively), a GPU parallel algorithm library for Rust, on top of [CubeCL](https://github.com/tracel-ai/cubecl).

## Experiments

This implementation was verified against standard shock-hydrodynamics benchmark problems, including the Sod shock tube, wall shock, Sjogreen rarefaction, and Noh implosion tests.

### Shocktube

[https://en.wikipedia.org/wiki/Sod_shock_tube](https://en.wikipedia.org/wiki/Sod_shock_tube)

![shocktube](workspace/plot/shocktube.jpeg)

### Wallshock

![wallshock](workspace/plot/wallshock.jpeg)

### Sjogreen

![sjogreen](workspace/plot/sjogreen.jpeg)

### Noh

![noh](workspace/plot/noh.jpeg)