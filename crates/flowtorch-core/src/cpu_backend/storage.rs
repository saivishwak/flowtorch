use crate::{
    backend::BackendStorage,
    error::StorageError,
    layout::Layout,
    ops::{BinaryOpT, UnaryOpT},
    DType,
};

use super::{error::*, utils, CpuDevice};
use utils::compare_vecs;

#[derive(Debug)]
pub enum ScalarStorage {
    F32(f32),
    F64(f64),
    U8(u8),
    U32(u32),
    I32(i32),
    I64(i64),
}

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
    pub fn from_scalar_storage_vec(
        data: Vec<ScalarStorage>,
    ) -> Result<CpuStorage, CpuStorageError> {
        if data.is_empty() {
            return Err(CpuStorageError::EmptyArray);
        }

        // Match on the type of the first element
        match &data[0] {
            ScalarStorage::F32(_) => {
                let converted_data: Vec<f32> = data
                    .into_iter()
                    .map(|v| match v {
                        ScalarStorage::F32(d) => d,
                        _ => unreachable!(), // Should not happen due to the match on first element
                    })
                    .collect();
                Ok(CpuStorage::F32(converted_data))
            }
            ScalarStorage::F64(_) => {
                let converted_data: Vec<f64> = data
                    .into_iter()
                    .map(|v| match v {
                        ScalarStorage::F64(d) => d,
                        _ => unreachable!(), // Should not happen due to the match on first element
                    })
                    .collect();
                Ok(CpuStorage::F64(converted_data))
            }
            ScalarStorage::U8(_) => {
                let converted_data: Vec<u8> = data
                    .into_iter()
                    .map(|v| match v {
                        ScalarStorage::U8(d) => d,
                        _ => unreachable!(), // Should not happen due to the match on first element
                    })
                    .collect();
                Ok(CpuStorage::U8(converted_data))
            }
            ScalarStorage::U32(_) => {
                let converted_data: Vec<u32> = data
                    .into_iter()
                    .map(|v| match v {
                        ScalarStorage::U32(d) => d,
                        _ => unreachable!(), // Should not happen due to the match on first element
                    })
                    .collect();
                Ok(CpuStorage::U32(converted_data))
            }
            ScalarStorage::I32(_) => {
                let converted_data: Vec<i32> = data
                    .into_iter()
                    .map(|v| match v {
                        ScalarStorage::I32(d) => d,
                        _ => unreachable!(), // Should not happen due to the match on first element
                    })
                    .collect();
                Ok(CpuStorage::I32(converted_data))
            }
            ScalarStorage::I64(_) => {
                let converted_data: Vec<i64> = data
                    .into_iter()
                    .map(|v| match v {
                        ScalarStorage::I64(d) => d,
                        _ => unreachable!(), // Should not happen due to the match on first element
                    })
                    .collect();
                Ok(CpuStorage::I64(converted_data))
            }
        }
    }

    fn get_raw(&self) -> CpuStorage {
        self.clone()
    }

    //TODO - Implement index_select
    #[allow(unused_variables)]
    pub(crate) fn index_select(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        dim: usize,
    ) -> Result<CpuStorage, StorageError> {
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
                                        return Err(CpuStorageError::ContiguousElementDtypeMismatch)
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

    fn to_dtype(&self, _layout: &Layout, dtype: DType) -> Result<Self, StorageError> {
        //TODO - Instead of iterating over everything, use the layout to get the strided elements only for perf
        // And also get a better way to handle these many dtypes
        match (self, dtype) {
            (CpuStorage::U8(data), DType::U32) => {
                let data: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                Ok(CpuStorage::U32(data))
            }
            (CpuStorage::U8(data), DType::I64) => {
                let data: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                Ok(CpuStorage::I64(data))
            }
            (CpuStorage::U8(data), DType::I32) => {
                let data: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                Ok(CpuStorage::I32(data))
            }
            (CpuStorage::U8(data), DType::F32) => {
                let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Ok(CpuStorage::F32(data))
            }
            (CpuStorage::U8(data), DType::F64) => {
                let data: Vec<f64> = data.iter().map(|&v| v as f64).collect();
                Ok(CpuStorage::F64(data))
            }
            (CpuStorage::U32(data), DType::U8) => {
                let data: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                Ok(CpuStorage::U8(data))
            }
            (CpuStorage::U32(data), DType::I64) => {
                let data: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                Ok(CpuStorage::I64(data))
            }
            (CpuStorage::U32(data), DType::I32) => {
                let data: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                Ok(CpuStorage::I32(data))
            }
            (CpuStorage::U32(data), DType::F32) => {
                let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Ok(CpuStorage::F32(data))
            }
            (CpuStorage::U32(data), DType::F64) => {
                let data: Vec<f64> = data.iter().map(|&v| v as f64).collect();
                Ok(CpuStorage::F64(data))
            }
            (CpuStorage::I32(data), DType::U8) => {
                let data: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                Ok(CpuStorage::U8(data))
            }
            (CpuStorage::I32(data), DType::U32) => {
                let data: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                Ok(CpuStorage::U32(data))
            }
            (CpuStorage::I32(data), DType::I64) => {
                let data: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                Ok(CpuStorage::I64(data))
            }
            (CpuStorage::I32(data), DType::F32) => {
                let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Ok(CpuStorage::F32(data))
            }
            (CpuStorage::I32(data), DType::F64) => {
                let data: Vec<f64> = data.iter().map(|&v| v as f64).collect();
                Ok(CpuStorage::F64(data))
            }
            (CpuStorage::I64(data), DType::U8) => {
                let data: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                Ok(CpuStorage::U8(data))
            }
            (CpuStorage::I64(data), DType::U32) => {
                let data: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                Ok(CpuStorage::U32(data))
            }
            (CpuStorage::I64(data), DType::I32) => {
                let data: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                Ok(CpuStorage::I32(data))
            }
            (CpuStorage::I64(data), DType::F32) => {
                let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Ok(CpuStorage::F32(data))
            }
            (CpuStorage::I64(data), DType::F64) => {
                let data: Vec<f64> = data.iter().map(|&v| v as f64).collect();
                Ok(CpuStorage::F64(data))
            }
            (CpuStorage::F32(data), DType::U8) => {
                let data: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                Ok(CpuStorage::U8(data))
            }
            (CpuStorage::F32(data), DType::U32) => {
                let data: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                Ok(CpuStorage::U32(data))
            }
            (CpuStorage::F32(data), DType::I64) => {
                let data: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                Ok(CpuStorage::I64(data))
            }
            (CpuStorage::F32(data), DType::I32) => {
                let data: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                Ok(CpuStorage::I32(data))
            }
            (CpuStorage::F32(data), DType::F64) => {
                let data: Vec<f64> = data.iter().map(|&v| v as f64).collect();
                Ok(CpuStorage::F64(data))
            }
            (CpuStorage::F64(data), DType::U8) => {
                let data: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                Ok(CpuStorage::U8(data))
            }
            (CpuStorage::F64(data), DType::U32) => {
                let data: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                Ok(CpuStorage::U32(data))
            }
            (CpuStorage::F64(data), DType::I64) => {
                let data: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                Ok(CpuStorage::I64(data))
            }
            (CpuStorage::F64(data), DType::I32) => {
                let data: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                Ok(CpuStorage::I32(data))
            }
            (CpuStorage::F64(data), DType::F32) => {
                let data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                Ok(CpuStorage::F32(data))
            }
            _ => Err(CpuStorageError::Custom("Data type conversion not found".to_string()).into()),
        }
    }

    fn unary_impl<U: UnaryOpT>(&self) -> Result<Self, StorageError> {
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
    fn equal(&self, rhs: &Self, lhs_offset: (usize, usize), rhs_offset: (usize, usize)) -> bool {
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
    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, StorageError> {
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
            _ => Err(CpuStorageError::MismatchDtype.into()),
        }
    }
}
