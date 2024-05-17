use crate::{cpu_backend::CpuStorage, storage::Storage, DType};
use ndarray::{ArrayD, IxDyn};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Storage, ()> {
        match self {
            Device::Cpu => {
                match dtype {
                    DType::F32 => {
                        let buffer = ArrayD::<f32>::zeros(IxDyn(shape)).into_raw_vec();
                        return Ok(Storage::Cpu(CpuStorage::F32(buffer)));
                    }
                    DType::F64 => {
                        let buffer = ArrayD::<f64>::zeros(IxDyn(shape)).into_raw_vec();
                        return Ok(Storage::Cpu(CpuStorage::F64(buffer)));
                    }
                    DType::U8 => todo!(),
                    DType::U32 => todo!(),
                    DType::I64 => todo!(),
                    DType::F16 => todo!(),
                };
            }
        }
    }
}
