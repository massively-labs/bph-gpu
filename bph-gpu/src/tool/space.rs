use super::*;

pub struct Cell {
    origin: (f32, f32, f32),
    space: (f32, f32, f32),
}

impl Cell {
    pub fn x_min(&self) -> f32 {
        self.origin.0
    }

    pub fn y_min(&self) -> f32 {
        self.origin.1
    }

    pub fn z_min(&self) -> f32 {
        self.origin.2
    }

    pub fn x_max(&self) -> f32 {
        self.x_min() + self.space.0
    }

    pub fn y_max(&self) -> f32 {
        self.y_min() + self.space.1
    }

    pub fn z_max(&self) -> f32 {
        self.z_min() + self.space.2
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone))]
pub struct Space {
    origin: (f32, f32, f32),
    space: (f32, f32, f32),
    dim: (u32, u32, u32),
}

impl Space {
    pub fn new(origin: (f32, f32, f32), space: (f32, f32, f32), dim: (u32, u32, u32)) -> Self {
        Self { origin, space, dim }
    }

    pub fn dim(&self) -> (u32, u32, u32) {
        self.dim
    }

    pub fn x_min(&self) -> f32 {
        self.origin.0
    }

    pub fn y_min(&self) -> f32 {
        self.origin.1
    }

    pub fn z_min(&self) -> f32 {
        self.origin.2
    }

    pub fn x_max(&self) -> f32 {
        self.x_min() + self.space.0 * self.dim.0 as f32
    }

    pub fn y_max(&self) -> f32 {
        self.y_min() + self.space.1 * self.dim.1 as f32
    }

    pub fn z_max(&self) -> f32 {
        self.z_min() + self.space.2 * self.dim.2 as f32
    }

    pub fn get_cell_at(&self, i: u32, j: u32, k: u32) -> Cell {
        assert!(i < self.dim.0);
        assert!(j < self.dim.1);
        assert!(k < self.dim.2);

        Cell {
            origin: (
                self.origin.0 + self.space.0 * i as f32,
                self.origin.1 + self.space.1 * j as f32,
                self.origin.2 + self.space.2 * k as f32,
            ),
            space: self.space,
        }
    }
}

#[cube]
impl Space {
    pub fn calc_cell_index_1d(&self, x: f32, y: f32, z: f32) -> u32 {
        let i = ((x - self.origin.0) / self.space.0) as u32;
        let j = ((y - self.origin.1) / self.space.1) as u32;
        let k = ((z - self.origin.2) / self.space.2) as u32;

        i + self.dim.0 * j + self.dim.0 * self.dim.1 * k
    }
}

pub struct CalcCellIndex1d;
#[cube]
impl<R: Runtime> UnaryOp<R, f32_3> for CalcCellIndex1d {
    type Env = Space;
    type Output = (u32,);
    fn apply(cell: Space, x: f32_3) -> (u32,) {
        (cell.calc_cell_index_1d(x.0, x.1, x.2),)
    }
}
