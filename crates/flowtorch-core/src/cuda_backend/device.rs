use std::sync::Arc;

pub use cudarc;

use crate::{
    backend::BackendDevice, cpu_backend::CpuStorage, dtype::WithDType, DType, DeviceError,
    DeviceErrorKind, Shape,
};

use super::{storage::CudaStorageSlice, CudaStorage};

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub(crate) ordinal: usize,
    pub(crate) device: Arc<cudarc::driver::CudaDevice>,
}

macro_rules! alloc_and_init {
    ($self:ident, $shape:ident, $dtype:ident, $init_val:expr, $err_kind:ident) => {{
        let shape_vec: Vec<usize> = $shape.into();
        let num_elements: usize = shape_vec.iter().product();
        match $dtype {
            DType::F64 => {
                let slice = $self.device.htod_copy(vec![$init_val as f64; num_elements]);
                if let Ok(s) = slice {
                    return Ok(CudaStorage {
                        device: $self.clone(),
                        slice: CudaStorageSlice::F64(s),
                    });
                }
                return Err(DeviceError::new(DeviceErrorKind::$err_kind));
            }
            DType::F32 => {
                let slice = $self.device.htod_copy(vec![$init_val as f32; num_elements]);
                if let Ok(s) = slice {
                    return Ok(CudaStorage {
                        device: $self.clone(),
                        slice: CudaStorageSlice::F32(s),
                    });
                }
                return Err(DeviceError::new(DeviceErrorKind::$err_kind));
            }
            DType::U8 => {
                let slice = $self.device.htod_copy(vec![$init_val as u8; num_elements]);
                if let Ok(s) = slice {
                    return Ok(CudaStorage {
                        device: $self.clone(),
                        slice: CudaStorageSlice::U8(s),
                    });
                }
                return Err(DeviceError::new(DeviceErrorKind::$err_kind));
            }
            DType::U32 => {
                let slice = $self.device.htod_copy(vec![$init_val as u32; num_elements]);
                if let Ok(s) = slice {
                    return Ok(CudaStorage {
                        device: $self.clone(),
                        slice: CudaStorageSlice::U32(s),
                    });
                }
                return Err(DeviceError::new(DeviceErrorKind::$err_kind));
            }
            DType::I32 => {
                let slice = $self.device.htod_copy(vec![$init_val as i32; num_elements]);
                if let Ok(s) = slice {
                    return Ok(CudaStorage {
                        device: $self.clone(),
                        slice: CudaStorageSlice::I32(s),
                    });
                }
                return Err(DeviceError::new(DeviceErrorKind::$err_kind));
            }
            DType::I64 => {
                let slice = $self.device.htod_copy(vec![$init_val as i64; num_elements]);
                if let Ok(s) = slice {
                    return Ok(CudaStorage {
                        device: $self.clone(),
                        slice: CudaStorageSlice::I64(s),
                    });
                }
                return Err(DeviceError::new(DeviceErrorKind::$err_kind));
            }
        }
    }};
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self, DeviceError> {
        let dev = cudarc::driver::CudaDevice::new(ordinal);
        if let Ok(d) = dev {
            Ok(Self { ordinal, device: d })
        } else {
            Err(DeviceError::new(crate::DeviceErrorKind::InitFail))
        }
    }

    fn as_str(&self) -> String {
        format!("Cuda({})", self.ordinal)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError> {
        alloc_and_init!(self, shape, dtype, 0, OnesFail)
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError> {
        alloc_and_init!(self, shape, dtype, 1, OnesFail)
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage, DeviceError> {
        let slice = match T::to_cpu_storage(data) {
            CpuStorage::U8(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::U8(s)
                } else {
                    return Err(DeviceError::new(DeviceErrorKind::CopyFail));
                }
            }
            CpuStorage::U32(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::U32(s)
                } else {
                    return Err(DeviceError::new(DeviceErrorKind::CopyFail));
                }
            }
            CpuStorage::I32(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::I32(s)
                } else {
                    return Err(DeviceError::new(DeviceErrorKind::CopyFail));
                }
            }
            CpuStorage::I64(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::I64(s)
                } else {
                    return Err(DeviceError::new(DeviceErrorKind::CopyFail));
                }
            }
            CpuStorage::F32(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::F32(s)
                } else {
                    return Err(DeviceError::new(DeviceErrorKind::CopyFail));
                }
            }
            CpuStorage::F64(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::F64(s)
                } else {
                    return Err(DeviceError::new(DeviceErrorKind::CopyFail));
                }
            }
        };

        Ok(CudaStorage {
            device: self.clone(),
            slice,
        })
    }

    fn storage_from_cpu_storage(
        &self,
        cpu_storage: &CpuStorage,
    ) -> Result<Self::Storage, DeviceError> {
        match cpu_storage {
            CpuStorage::F32(data) => self.storage_from_slice(data),
            CpuStorage::U8(data) => self.storage_from_slice(data),
            CpuStorage::U32(data) => self.storage_from_slice(data),
            CpuStorage::I32(data) => self.storage_from_slice(data),
            CpuStorage::I64(data) => self.storage_from_slice(data),
            CpuStorage::F64(data) => self.storage_from_slice(data),
        }
    }
}
