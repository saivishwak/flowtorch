use crate::{backend::BackendDevice, shape::Shape, DType, DeviceError};

use super::CpuStorage;

#[derive(Debug, Clone)]
pub struct CpuDevice;

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn new(_: usize) -> Result<Self, DeviceError> {
        Ok(Self)
    }

    fn as_str(&self) -> String {
        String::from("Cpu")
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError> {
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

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError> {
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

    fn storage_from_slice<T: crate::dtype::WithDType>(
        &self,
        data: &[T],
    ) -> Result<Self::Storage, DeviceError> {
        Ok(T::to_cpu_storage(data))
    }

    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &CpuStorage,
    ) -> Result<Self::Storage, DeviceError> {
        Ok(cpu_storage.clone())
    }
}
