use ndarray::{ArrayD, IxDyn};

use crate::{shape::Shape, DType};

pub enum CpuStorage {
    U8(ArrayD<u8>),
    U32(ArrayD<u32>),
    I64(ArrayD<i64>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

pub struct CpuDevice;

impl CpuDevice {
    pub fn zeros(shape: &Shape, dtype: DType) -> Result<CpuStorage, ()> {
        match dtype {
            DType::F32 => {
                let buffer = ArrayD::<f32>::zeros(IxDyn(shape));
                return Ok(CpuStorage::F32(buffer));
            }
            DType::F64 => {
                let buffer = ArrayD::<f64>::zeros(IxDyn(shape));
                return Ok(CpuStorage::F64(buffer));
            }
            DType::U8 => {
                let buffer = ArrayD::<u8>::zeros(IxDyn(shape));
                return Ok(CpuStorage::U8(buffer));
            }
            DType::U32 => {
                let buffer = ArrayD::<u32>::zeros(IxDyn(shape));
                return Ok(CpuStorage::U32(buffer));
            }
            DType::I64 => {
                let buffer = ArrayD::<i64>::zeros(IxDyn(shape));
                return Ok(CpuStorage::I64(buffer));
            }
        };
    }
}
