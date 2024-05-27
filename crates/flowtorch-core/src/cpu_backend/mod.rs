mod error;

use crate::{shape::Shape, storage::BaseStorage, DType, Error};
pub use error::*;

#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    //All the DataTypes should be same for a sequence of storages.
    pub fn concat(storages: &[CpuStorage]) -> Result<CpuStorage, CpuStorageError> {
        let storage0 = &storages[0];
        let s = match storage0 {
            Self::U8(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U8(s) => Ok(s.as_slice()),
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
                        Self::U8(storages_concatenated)
                    }
                    Err(e) => return Err(e),
                }
            }
            Self::U32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U32(s) => Ok(s.as_slice()),
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
                        Self::U32(storages_concatenated)
                    }
                    Err(e) => return Err(e),
                }
            }
            Self::I64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I64(s) => Ok(s.as_slice()),
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
                        Self::I64(storages_concatenated)
                    }
                    Err(e) => return Err(e),
                }
            }
            Self::F32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F32(s) => Ok(s.as_slice()),
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
                        Self::F32(storages_concatenated)
                    }
                    Err(e) => return Err(e),
                }
            }
            Self::F64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F64(s) => Ok(s.as_slice()),
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
                        Self::F64(storages_concatenated)
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        Ok(s)
    }

    fn get_raw(&self) -> Box<&CpuStorage> {
        return Box::new(self);
    }
}

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
        }
    }
}
