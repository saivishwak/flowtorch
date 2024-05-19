use crate::{device::NdArray, shape::Shape, DType};

#[derive(Debug)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    pub fn concat(storages: &[CpuStorage]) -> Result<CpuStorage, ()> {
        let storage0 = &storages[0];
        let s = match storage0 {
            Self::U8(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U8(s) => Ok(s.as_slice()),
                        _ => Err(()),
                    })
                    .collect::<Result<Vec<_>, ()>>()?
                    .concat();
                Self::U8(storages)
            }
            Self::U32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U32(s) => Ok(s.as_slice()),
                        _ => Err(()),
                    })
                    .collect::<Result<Vec<_>, ()>>()?
                    .concat();
                Self::U32(storages)
            }
            Self::I64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I64(s) => Ok(s.as_slice()),
                        _ => Err(()),
                    })
                    .collect::<Result<Vec<_>, ()>>()?
                    .concat();
                Self::I64(storages)
            }
            Self::F32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F32(s) => Ok(s.as_slice()),
                        _ => Err(()),
                    })
                    .collect::<Result<Vec<_>, ()>>()?
                    .concat();
                Self::F32(storages)
            }
            Self::F64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F64(s) => Ok(s.as_slice()),
                        _ => Err(()),
                    })
                    .collect::<Result<Vec<_>, ()>>()?
                    .concat();
                Self::F64(storages)
            }
            _ => {
                return Err(());
            }
        };
        Ok(s)
    }
}
pub struct CpuDevice;

impl CpuDevice {
    pub fn zeros(shape: &Shape, dtype: DType) -> Result<CpuStorage, ()> {
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

    pub fn ones(shape: &Shape, dtype: DType) -> Result<CpuStorage, ()> {
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

    pub fn from_array<D: NdArray>(array: D) -> Result<CpuStorage, ()> {
        Ok(array.to_cpu_storage())
    }
}
