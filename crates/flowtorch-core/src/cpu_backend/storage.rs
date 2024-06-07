use crate::{
    backend::BackendStorage,
    layout::Layout,
    ops::{BinaryOpT, UnaryOpT},
    DType, Error,
};

use super::{error::*, utils, CpuDevice};
use utils::compare_vecs;

#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    fn get_raw(&self) -> CpuStorage {
        return self.clone();
    }

    pub(crate) fn equal(
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
    pub(crate) fn index_select(
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
            pub(crate) fn concat(storages: &[CpuStorage]) -> Result<CpuStorage, CpuStorageError> {
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

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn get_cpu_storage(&self) -> CpuStorage {
        self.get_raw()
    }

    fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::F64(_) => DType::F64,
        }
    }

    fn device(&self) -> &Self::Device {
        &CpuDevice
    }
}
