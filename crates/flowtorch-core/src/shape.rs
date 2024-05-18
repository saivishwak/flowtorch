#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    // Method to create a new Shape
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    // Method to get the dimensions
    pub fn dims(&self) -> &Vec<usize> {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    // Method to calculate the number of elements (product of dimensions)
    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }

    pub fn strides_contiguous(&self) -> Vec<usize> {
        let shape = &self.0;
        let mut strides: Vec<usize> = vec![0; shape.len()];

        if shape.len() == 0 {
            return strides;
        }

        //as this is first time insertion, the last element is 1
        strides[shape.len() - 1] = 1;

        //Early return as 1D array has strides as 1
        if shape.len() == 1 {
            return strides;
        }

        for i in (0..=shape.len() - 2).rev() {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        return strides;
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

impl From<&Shape> for Vec<usize> {
    fn from(shape: &Shape) -> Self {
        shape.0.clone()
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

impl From<(usize,)> for Shape {
    fn from(d1: (usize,)) -> Self {
        Self(vec![d1.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(d12: (usize, usize)) -> Self {
        Self(vec![d12.0, d12.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(d123: (usize, usize, usize)) -> Self {
        Self(vec![d123.0, d123.1, d123.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(d1234: (usize, usize, usize, usize)) -> Self {
        Self(vec![d1234.0, d1234.1, d1234.2, d1234.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(d12345: (usize, usize, usize, usize, usize)) -> Self {
        Self(vec![d12345.0, d12345.1, d12345.2, d12345.3, d12345.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(d123456: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self(vec![
            d123456.0, d123456.1, d123456.2, d123456.3, d123456.4, d123456.5,
        ])
    }
}
