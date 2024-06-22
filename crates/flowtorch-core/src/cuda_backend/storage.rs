use std::usize;

pub use cudarc;
use cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice, LaunchConfig};
use flowcuda_kernels::CAST;

use crate::error::StorageError;
use crate::layout::Layout;
use crate::ops::UnaryOpT;
use crate::{backend::BackendStorage, cpu_backend::CpuStorage, ops::BinaryOpT, CudaDevice, DType};

use super::ops::{Pair1Runner, Pair2Runner};

#[derive(Debug)]
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I32(CudaSlice<i32>),
    I64(CudaSlice<i64>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl CudaStorageSlice {
    fn get_storage_pointer(&self) -> u64 {
        let slice = match self {
            Self::U8(data) => *data.slice(0..data.len()).device_ptr(),
            Self::U32(data) => *data.slice(0..data.len()).device_ptr(),
            Self::I32(data) => *data.slice(0..data.len()).device_ptr(),
            Self::I64(data) => *data.slice(0..data.len()).device_ptr(),
            Self::F32(data) => *data.slice(0..data.len()).device_ptr(),
            Self::F64(data) => *data.slice(0..data.len()).device_ptr(),
        };
        slice
    }
}

type S = CudaStorageSlice;

#[derive(Debug)]
pub struct CudaStorage {
    pub device: CudaDevice,
    pub slice: S,
}

impl CudaStorage {
    pub fn new(device: CudaDevice, slice: S) -> Self {
        Self { device, slice }
    }
}

impl BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn get_cpu_storage(&self) -> CpuStorage {
        let slice = &self.slice;
        match slice {
            CudaStorageSlice::F32(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                CpuStorage::F32(b_host)
            }
            CudaStorageSlice::F64(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                CpuStorage::F64(b_host)
            }
            CudaStorageSlice::I64(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                CpuStorage::I64(b_host)
            }
            CudaStorageSlice::I32(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                CpuStorage::I32(b_host)
            }
            CudaStorageSlice::U8(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                CpuStorage::U8(b_host)
            }
            CudaStorageSlice::U32(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                CpuStorage::U32(b_host)
            }
        }
    }

    fn dtype(&self) -> DType {
        let slice = &self.slice;
        match slice {
            S::U8(_) => DType::U8,
            S::U32(_) => DType::U32,
            S::I32(_) => DType::I32,
            S::I64(_) => DType::I64,
            S::F32(_) => DType::F32,
            S::F64(_) => DType::F64,
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self, StorageError> {
        let slice = &self.slice;
        let len: usize = layout.shape().num_elements();
        let slice_ptr = slice.get_storage_pointer();
        let func = self.device.get_and_load_kernal_func(
            format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str()).as_str(),
            CAST,
        )?;
        match dtype {
            DType::F32 => {
                let out = self.device.alloc::<f32>(len)?;
                let launch_config = LaunchConfig::for_num_elems(len as u32);
                let params: (usize, u64, &CudaSlice<f32>) = (len, slice_ptr, &out);
                let _ = self
                    .device
                    .launch_function::<(usize, u64, &CudaSlice<f32>)>(
                        func,
                        launch_config,
                        params,
                    )?;
                return Ok(CudaStorage::new(
                    self.device().clone(),
                    CudaStorageSlice::F32(out),
                ));
            }
            DType::F64 => {
                let out = self.device.alloc::<f64>(len)?;
                let launch_config = LaunchConfig::for_num_elems(len as u32);
                let params: (usize, u64, &CudaSlice<f64>) = (len, slice_ptr, &out);
                let _ = self
                    .device
                    .launch_function::<(usize, u64, &CudaSlice<f64>)>(
                        func,
                        launch_config,
                        params,
                    )?;
                return Ok(CudaStorage::new(
                    self.device().clone(),
                    CudaStorageSlice::F64(out),
                ));
            }
            DType::I64 => {
                let out = self.device.alloc::<i64>(len)?;
                let launch_config = LaunchConfig::for_num_elems(len as u32);
                let params: (usize, u64, &CudaSlice<i64>) = (len, slice_ptr, &out);
                let _ = self
                    .device
                    .launch_function::<(usize, u64, &CudaSlice<i64>)>(
                        func,
                        launch_config,
                        params,
                    )?;
                return Ok(CudaStorage::new(
                    self.device().clone(),
                    CudaStorageSlice::I64(out),
                ));
            }
            DType::I32 => {
                let out = self.device.alloc::<i32>(len)?;
                let launch_config = LaunchConfig::for_num_elems(len as u32);
                let params: (usize, u64, &CudaSlice<i32>) = (len, slice_ptr, &out);
                let _ = self
                    .device
                    .launch_function::<(usize, u64, &CudaSlice<i32>)>(
                        func,
                        launch_config,
                        params,
                    )?;
                return Ok(CudaStorage::new(
                    self.device().clone(),
                    CudaStorageSlice::I32(out),
                ));
            }
            DType::U32 => {
                let out = self.device.alloc::<u32>(len)?;
                let launch_config = LaunchConfig::for_num_elems(len as u32);
                let params: (usize, u64, &CudaSlice<u32>) = (len, slice_ptr, &out);
                let _ = self
                    .device
                    .launch_function::<(usize, u64, &CudaSlice<u32>)>(
                        func,
                        launch_config,
                        params,
                    )?;
                return Ok(CudaStorage::new(
                    self.device().clone(),
                    CudaStorageSlice::U32(out),
                ));
            }
            DType::U8 => {
                let out = self.device.alloc::<u8>(len)?;
                let launch_config = LaunchConfig::for_num_elems(len as u32);
                let params: (usize, u64, &CudaSlice<u8>) = (len, slice_ptr, &out);
                let _ = self
                    .device
                    .launch_function::<(usize, u64, &CudaSlice<u8>)>(func, launch_config, params)?;
                return Ok(CudaStorage::new(
                    self.device().clone(),
                    CudaStorageSlice::U8(out),
                ));
            }
        }
    }

    fn unary_impl<U: UnaryOpT>(&self) -> Result<Self, StorageError> {
        let slice = U::V.run_op(self.device().clone(), self)?;
        Ok(Self {
            device: self.device().clone(),
            slice,
        })
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, StorageError> {
        let slice = B::V.run_op(self.device().clone(), self, rhs)?;
        Ok(Self {
            device: self.device().clone(),
            slice,
        })
    }

    fn equal(&self, rhs: &Self, self_offset: (usize, usize), other_offset: (usize, usize)) -> bool {
        let lhs_cpu_storage = self.get_cpu_storage();
        let rhs_cpu_storage = rhs.get_cpu_storage();
        lhs_cpu_storage.equal(&rhs_cpu_storage, self_offset, other_offset)
    }
}
