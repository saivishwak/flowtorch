use std::{fmt::Debug, sync::Arc};

pub use cudarc;
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use flowcuda_kernels::FILL;

use crate::{
    backend::BackendDevice, cpu_backend::CpuStorage, dtype::WithDType, error::DeviceError, DType,
    Shape,
};

use super::{storage::CudaStorageSlice, CudaDeviceError, CudaStorage};

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub(crate) ordinal: usize,
    pub(crate) device: Arc<cudarc::driver::CudaDevice>,
}

macro_rules! allocate_and_fill {
    ($self:expr, $dtype:ty, $DType: ident, $fill_val:expr, $num_elements:expr, $func:expr) => {{
        let data = unsafe { $self.device.alloc::<$dtype>($num_elements) };
        if let Err(e) = data {
            return Err(CudaDeviceError::AllocFail(Some(format!("{}", e))).into());
        }
        let data = data.unwrap();
        let launch_config = LaunchConfig::for_num_elems($num_elements as u32);
        let params = (&data, $fill_val, $num_elements);
        match unsafe { $func.launch(launch_config, params) } {
            Ok(_) => Ok(CudaStorage {
                device: $self.clone(),
                slice: CudaStorageSlice::$DType(data),
            }),
            Err(e) => return Err(CudaDeviceError::AllocFail(Some(format!("{}", e))).into()),
        }
    }};
}

impl CudaDevice {
    pub fn get_and_load_kernal_func(
        &self,
        module_name: &str,
        ptx: &'static str,
    ) -> Result<cudarc::driver::CudaFunction, DeviceError> {
        let dev: &Arc<cudarc::driver::CudaDevice> = &self.device;
        if !dev.has_func(module_name, module_name) {
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            if let Err(e) = dev.load_ptx(ptx.into(), module_name, &[static_module_name]) {
                let e_str = format!("{}", e);
                return Err(CudaDeviceError::AllocFail(Some(e_str)).into());
            }
        }
        dev.get_func(module_name, module_name)
            .ok_or(CudaDeviceError::MissingKernel(module_name.to_string()).into())
    }

    pub fn alloc<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        numel: usize,
    ) -> Result<CudaSlice<T>, DeviceError> {
        let data = unsafe { self.device.alloc::<T>(numel) };
        if let Err(e) = data {
            return Err(CudaDeviceError::AllocFail(Some(format!("{}", e))).into());
        }
        Ok(data.unwrap())
    }

    fn const_alloc<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        fill_val: T,
        shape: &Shape,
    ) -> Result<CudaStorage, DeviceError> {
        let shape_vec: Vec<usize> = shape.into();
        let num_elements: usize = shape_vec.iter().product();
        let func =
            self.get_and_load_kernal_func(format!("fill_{}", T::dtype().as_str()).as_str(), FILL)?;
        match T::dtype() {
            DType::U8 => allocate_and_fill!(self, u8, U8, fill_val, num_elements, func),
            DType::U32 => allocate_and_fill!(self, u32, U32, fill_val, num_elements, func),
            DType::I64 => allocate_and_fill!(self, i64, I64, fill_val, num_elements, func),
            DType::I32 => allocate_and_fill!(self, i32, I32, fill_val, num_elements, func),
            DType::F32 => allocate_and_fill!(self, f32, F32, fill_val, num_elements, func),
            DType::F64 => allocate_and_fill!(self, f64, F64, fill_val, num_elements, func),
        }
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self, DeviceError> {
        let dev = cudarc::driver::CudaDevice::new(ordinal);
        if let Ok(d) = dev {
            Ok(Self { ordinal, device: d })
        } else {
            Err(DeviceError::InitFail)
        }
    }

    fn as_str(&self) -> String {
        format!("Cuda({})", self.ordinal)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError> {
        match dtype {
            DType::F32 => self.const_alloc(0.0f32, shape),
            DType::F64 => self.const_alloc(0.0f64, shape),
            DType::U8 => self.const_alloc(0u8, shape),
            DType::U32 => self.const_alloc(0u32, shape),
            DType::I32 => self.const_alloc(0i32, shape),
            DType::I64 => self.const_alloc(0i64, shape),
        }
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage, DeviceError> {
        match dtype {
            DType::F32 => self.const_alloc(1.0f32, shape),
            DType::F64 => self.const_alloc(1.0f64, shape),
            DType::U8 => self.const_alloc(1u8, shape),
            DType::U32 => self.const_alloc(1u32, shape),
            DType::I32 => self.const_alloc(1i32, shape),
            DType::I64 => self.const_alloc(1i64, shape),
        }
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage, DeviceError> {
        let slice = match T::to_cpu_storage(data) {
            CpuStorage::U8(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::U8(s)
                } else {
                    return Err(DeviceError::CopyFail);
                }
            }
            CpuStorage::U32(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::U32(s)
                } else {
                    return Err(DeviceError::CopyFail);
                }
            }
            CpuStorage::I32(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::I32(s)
                } else {
                    return Err(DeviceError::CopyFail);
                }
            }
            CpuStorage::I64(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::I64(s)
                } else {
                    return Err(DeviceError::CopyFail);
                }
            }
            CpuStorage::F32(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::F32(s)
                } else {
                    return Err(DeviceError::CopyFail);
                }
            }
            CpuStorage::F64(data) => {
                let slice = self.device.htod_sync_copy(&data);
                if let Ok(s) = slice {
                    CudaStorageSlice::F64(s)
                } else {
                    return Err(DeviceError::CopyFail);
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
