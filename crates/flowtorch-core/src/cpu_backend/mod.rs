mod error;

use std::ops::{Add, Mul};

use crate::{layout::Layout, shape::Shape, storage::BaseStorage, DType, Error};
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
            (Self::F32(lhs_data), Self::F32(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            (Self::F64(lhs_data), Self::F64(rhs_data)) => {
                compare_vecs(lhs_data, rhs_data, lhs_offset, rhs_offset)
            }
            _ => false,
        }
    }

    pub(super) fn add(&self, rhs: &Self) -> Result<CpuStorage, Error> {
        match (self, rhs) {
            (Self::U8(lhs_data), Self::U8(rhs_data)) => {
                return Ok(CpuStorage::U8(add_vec(lhs_data, rhs_data)));
            }
            (Self::U32(lhs_data), Self::U32(rhs_data)) => {
                return Ok(CpuStorage::U32(add_vec(lhs_data, rhs_data)));
            }
            (Self::I64(lhs_data), Self::I64(rhs_data)) => {
                return Ok(CpuStorage::I64(add_vec(lhs_data, rhs_data)));
            }
            (Self::F32(lhs_data), Self::F32(rhs_data)) => {
                return Ok(CpuStorage::F32(add_vec(lhs_data, rhs_data)));
            }
            (Self::F64(lhs_data), Self::F64(rhs_data)) => {
                return Ok(CpuStorage::F64(add_vec(lhs_data, rhs_data)));
            }
            _ => {
                return Err(Error::Unknown);
            }
        }
    }

    pub(super) fn mul(&self, rhs: &Self) -> Result<CpuStorage, Error> {
        match (self, rhs) {
            (Self::U8(lhs_data), Self::U8(rhs_data)) => {
                return Ok(CpuStorage::U8(mul_vec(lhs_data, rhs_data)));
            }
            (Self::U32(lhs_data), Self::U32(rhs_data)) => {
                return Ok(CpuStorage::U32(mul_vec(lhs_data, rhs_data)));
            }
            (Self::I64(lhs_data), Self::I64(rhs_data)) => {
                return Ok(CpuStorage::I64(mul_vec(lhs_data, rhs_data)));
            }
            (Self::F32(lhs_data), Self::F32(rhs_data)) => {
                return Ok(CpuStorage::F32(mul_vec(lhs_data, rhs_data)));
            }
            (Self::F64(lhs_data), Self::F64(rhs_data)) => {
                return Ok(CpuStorage::F64(mul_vec(lhs_data, rhs_data)));
            }
            _ => {
                return Err(Error::Unknown);
            }
        }
    }

    //TODO - Implement index_select
    pub(super) fn index_select(
        &self,
        _rhs: &Self,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
        _dim: usize,
    ) -> Result<CpuStorage, Error> {
        todo!()
    }
}

impl BaseStorage for CpuStorage {
    fn cpu_get_raw(&self) -> Box<&CpuStorage> {
        self.get_raw()
    }
}

// Helper function to compare vectors of any type
fn compare_vecs<T: PartialEq>(
    vec1: &Vec<T>,
    vec2: &Vec<T>,
    vec1_offset: (usize, usize),
    vec2_offset: (usize, usize),
) -> bool {
    let vec1_start = vec1_offset.0;
    let vec1_end = vec1_start + vec1_offset.1;
    let vec2_start = vec2_offset.0;
    let vec2_end = vec2_start + vec2_offset.1;
    if vec1_end - vec1_start != vec2_end - vec2_start {
        return false;
    }
    &vec1[vec1_start..vec1_end] == &vec2[vec2_start..vec2_end]
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

fn add_vec<T>(vec1: &Vec<T>, vec2: &Vec<T>) -> Vec<T>
where
    T: Add<Output = T> + Copy,
{
    vec1.iter().zip(vec2.iter()).map(|(a, b)| *a + *b).collect()
}

fn mul_vec<T>(vec1: &Vec<T>, vec2: &Vec<T>) -> Vec<T>
where
    T: Mul<Output = T> + Copy,
{
    vec1.iter().zip(vec2.iter()).map(|(a, b)| *a * *b).collect()
}
