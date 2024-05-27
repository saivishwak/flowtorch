mod error;
use crate::{
    cpu_backend::{CpuStorage, CpuStorageError},
    dtype::WithDType,
    Shape,
};

pub use error::NDArrayError;

pub trait NdArray {
    fn shape(&self) -> Result<Shape, NDArrayError>;
    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError>;
}

//Scalar value, hence empty shape
impl<S: WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape, NDArrayError> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError> {
        Ok(S::to_cpu_storage(&[*self]))
    }
}

//1D dymanic length Array
impl<S: WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape, NDArrayError> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError> {
        Ok(S::to_cpu_storage(self))
    }
}

// 1D fixed length Array
impl<S: WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape, NDArrayError> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError> {
        Ok(S::to_cpu_storage(self.as_slice()))
    }
}

//2D Array
impl<S: WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape, NDArrayError> {
        Ok(Shape::from((M, N)))
    }
    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError> {
        Ok(S::to_cpu_storage(self.concat().as_slice()))
    }
}

//3D Array
impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[S; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape, NDArrayError> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2]);
            }
        }
        Ok(S::to_cpu_storage(vec.as_slice()))
    }
}

//4D Array
impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> NdArray
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape, NDArrayError> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError> {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3]);
                }
            }
        }
        Ok(S::to_cpu_storage(vec.as_slice()))
    }
}

//Recursive implementation for Vectors
impl<S: NdArray> NdArray for Vec<S> {
    fn shape(&self) -> Result<Shape, NDArrayError> {
        if self.is_empty() {
            return Err(NDArrayError::new(error::NDArrayErrorKind::EmptyShape));
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                //two elements have different shapes {shape:?} {shape0:?}
                return Err(NDArrayError::new(error::NDArrayErrorKind::ShapeMismatch));
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage, CpuStorageError> {
        let storages = self
            .iter()
            .map(|v| v.to_cpu_storage())
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        CpuStorage::concat(storages.as_slice())
    }
}
