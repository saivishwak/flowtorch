pub use cudarc;
use cudarc::driver::CudaSlice;

use crate::{backend::BackendStorage, cpu_backend::CpuStorage, CudaDevice, DType};

#[derive(Debug)]
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I32(CudaSlice<i32>),
    I64(CudaSlice<i64>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

#[derive(Debug)]
pub struct CudaStorage {
    pub device: CudaDevice,
    pub slice: CudaStorageSlice,
}

impl CudaStorage {
    pub fn new(device: CudaDevice, slice: CudaStorageSlice) -> Self {
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
                return CpuStorage::F32(b_host);
            }
            CudaStorageSlice::F64(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                return CpuStorage::F64(b_host);
            }
            CudaStorageSlice::I64(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                return CpuStorage::I64(b_host);
            }
            CudaStorageSlice::I32(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                return CpuStorage::I32(b_host);
            }
            CudaStorageSlice::U8(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                return CpuStorage::U8(b_host);
            }
            CudaStorageSlice::U32(d) => {
                let b_host = self.device.device.dtoh_sync_copy(d).unwrap();
                return CpuStorage::U32(b_host);
            }
        }
    }

    fn dtype(&self) -> DType {
        let slice = &self.slice;
        match slice {
            CudaStorageSlice::U8(_) => DType::U8,
            CudaStorageSlice::U32(_) => DType::U32,
            CudaStorageSlice::I32(_) => DType::I32,
            CudaStorageSlice::I64(_) => DType::I64,
            CudaStorageSlice::F32(_) => DType::F32,
            CudaStorageSlice::F64(_) => DType::F64,
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }
}
