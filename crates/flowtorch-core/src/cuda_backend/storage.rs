use std::usize;

pub use cudarc;
use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use flowcuda_kernels::CAST;

use crate::error::StorageError;
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

    fn to_dtype(
        &self,
        _layout: &crate::layout::Layout,
        dtype: DType,
    ) -> Result<Self, StorageError> {
        if self.dtype() == dtype {
            // return Ok(self.clone());
        }
        let slice = &self.slice;
        match (slice, dtype) {
            (CudaStorageSlice::F64(slice), DType::F32) => {
                let data = self.device.alloc::<f32>(slice.len());
                match data {
                    Ok(data) => {
                        let func = self
                            .device
                            .get_and_load_kernal_func("cast_f64_f32", CAST)
                            .unwrap();
                        let launch_config = LaunchConfig::for_num_elems(slice.len() as u32);
                        let params = (&data, slice.len());
                        match unsafe { func.launch(launch_config, params) } {
                            Ok(_) => {
                                return Ok(CudaStorage::new(
                                    self.device().clone(),
                                    CudaStorageSlice::F32(data),
                                ))
                            }
                            Err(_) => return Err(StorageError::Unknown),
                        }
                    }
                    Err(_e) => {
                        return Err(StorageError::Unknown);
                    }
                }
            }
            _ => {
                todo!()
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
