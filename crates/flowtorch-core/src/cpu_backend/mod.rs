mod error;
mod utils;

use crate::{
    layout::Layout,
    ops::{BinaryOpT, UnaryOpT},
    shape::Shape,
    storage::BaseStorage,
    DType, Error,
};
pub use error::*;
use utils::compare_vecs;

#[derive(Debug)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    fn get_raw(&self) -> Box<&CpuStorage> {
        return Box::new(self);
    }

    pub(super) fn equal(
        &self,
        rhs: &Self,
        lhs_offset: (usize, usize),
        rhs_offset: (usize, usize),
    ) -> bool {
        match (self, rhs) {
            (Self::U8(lhs_data), Self::U8(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            (Self::U32(lhs_data), Self::U32(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            (Self::I64(lhs_data), Self::I64(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            (Self::I32(lhs_data), Self::I32(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            (Self::F32(lhs_data), Self::F32(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            (Self::F64(lhs_data), Self::F64(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            _ => false,
        }
    }

    #[allow(unused_variables)]
    pub(crate) fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, Error> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| B::f32(*lhs, *rhs))
                    .collect();
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| B::f64(*lhs, *rhs))
                    .collect();
                Ok(Self::F64(data))
            }
            (Self::U8(lhs), Self::U8(rhs)) => {
                let data = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| B::u8(*lhs, *rhs))
                    .collect();
                Ok(Self::U8(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| B::u32(*lhs, *rhs))
                    .collect();
                Ok(Self::U32(data))
            }
            (Self::I32(lhs), Self::I32(rhs)) => {
                let data = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| B::i32(*lhs, *rhs))
                    .collect();
                Ok(Self::I32(data))
            }
            (Self::I64(lhs), Self::I64(rhs)) => {
                let data = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| B::i64(*lhs, *rhs))
                    .collect();
                Ok(Self::I64(data))
            }
            _ => {
                return Err(Error::Unknown);
            }
        }
    }

    pub(crate) fn unary_impl<U: UnaryOpT>(&self) -> Result<Self, Error> {
        match self {
            Self::F32(lhs) => {
                let data = lhs.iter().map(|v| U::f32(*v)).collect();
                Ok(Self::F32(data))
            }
            Self::F64(lhs) => {
                let data = lhs.iter().map(|v| U::f64(*v)).collect();
                Ok(Self::F64(data))
            }
            Self::U8(lhs) => {
                let data = lhs.iter().map(|v| U::u8(*v)).collect();
                Ok(Self::U8(data))
            }
            Self::U32(lhs) => {
                let data = lhs.iter().map(|v| U::u32(*v)).collect();
                Ok(Self::U32(data))
            }
            Self::I32(lhs) => {
                let data = lhs.iter().map(|v| U::i32(*v)).collect();
                Ok(Self::I32(data))
            }
            Self::I64(lhs) => {
                let data = lhs.iter().map(|v| U::i64(*v)).collect();
                Ok(Self::I64(data))
            }
        }
    }

    //TODO - Implement index_select
    #[allow(unused_variables)]
    pub(super) fn index_select(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        dim: usize,
    ) -> Result<CpuStorage, Error> {
        todo!()
    }
}

macro_rules! concat_impl {
    ($($variant:ident),*) => {
        impl CpuStorage {
            //All the DataTypes should be same for a sequence of storages.
            pub(super) fn concat(storages: &[CpuStorage]) -> Result<CpuStorage, CpuStorageError> {
                let storage0 = &storages[0];
                match storage0 {
                    $(
                        Self::$variant(_) => {
                            let storages = storages
                                .iter()
                                .map(|s| match s {
                                    Self::$variant(s) => Ok(s.as_slice()),
                                    _ => {
                                        return Err(CpuStorageError::new(
                                            CpuStorageErrorKind::ContiguousElementDtypeMismatch,
                                        ))
                                    }
                                })
                                .collect::<Result<Vec<_>, CpuStorageError>>();
                            match storages {
                                Ok(storages_unwrapped) => {
                                    let storages_concatenated = storages_unwrapped.concat();
                                    Ok(Self::$variant(storages_concatenated))
                                }
                                Err(e) => Err(e),
                            }
                        }
                    )*
                }
            }
        }
    }
}

// Applying the macro to generate the `concat` implementation for the variants
concat_impl!(U8, U32, I32, I64, F32, F64);

impl BaseStorage for CpuStorage {
    fn cpu_get_raw(&self) -> Box<&CpuStorage> {
        self.get_raw()
    }
}

pub struct CpuDevice;

impl CpuDevice {
    pub fn zeros(shape: &Shape, dtype: DType) -> Result<CpuStorage, Error> {
        let shape_vec: Vec<usize> = shape.into();
        let num_elements = shape_vec.iter().product();
        match dtype {
            DType::F32 => Ok(CpuStorage::F32(vec![0f32; num_elements])),
            DType::F64 => Ok(CpuStorage::F64(vec![0f64; num_elements])),
            DType::U8 => Ok(CpuStorage::U8(vec![0u8; num_elements])),
            DType::U32 => Ok(CpuStorage::U32(vec![0u32; num_elements])),
            DType::I64 => Ok(CpuStorage::I64(vec![0i64; num_elements])),
            DType::I32 => Ok(CpuStorage::I32(vec![0i32; num_elements])),
        }
    }

    pub fn ones(shape: &Shape, dtype: DType) -> Result<CpuStorage, Error> {
        let shape_vec: Vec<usize> = shape.into();
        let num_elements = shape_vec.iter().product();
        match dtype {
            DType::F32 => Ok(CpuStorage::F32(vec![1f32; num_elements])),
            DType::F64 => Ok(CpuStorage::F64(vec![1f64; num_elements])),
            DType::U8 => Ok(CpuStorage::U8(vec![1u8; num_elements])),
            DType::U32 => Ok(CpuStorage::U32(vec![1u32; num_elements])),
            DType::I64 => Ok(CpuStorage::I64(vec![1i64; num_elements])),
            DType::I32 => Ok(CpuStorage::I32(vec![1i32; num_elements])),
        }
    }
}
