use ndarray::{ArrayD, IxDyn};

use crate::{shape::Shape, DType};

pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub struct CpuDevice;

impl CpuDevice {
    pub fn zeros(shape: &Shape, dtype: DType) -> Result<CpuStorage, ()> {
        match dtype {
            DType::F32 => {
                let buffer = ArrayD::<f32>::zeros(IxDyn(shape)).into_raw_vec();
                return Ok(CpuStorage::F32(buffer));
            }
            DType::F64 => {
                let buffer = ArrayD::<f64>::zeros(IxDyn(shape)).into_raw_vec();
                return Ok(CpuStorage::F64(buffer));
            }
            DType::U8 => todo!(),
            DType::U32 => todo!(),
            DType::I64 => todo!(),
            DType::F16 => todo!(),
        };
    }
}
