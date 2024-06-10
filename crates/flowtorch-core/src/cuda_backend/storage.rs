#![allow(unused_imports)]
pub use cudarc;
use cudarc::driver::{CudaFunction, LaunchConfig};
use cudarc::driver::{CudaSlice, LaunchAsync};
use cudarc::nvrtc::Ptx;

use crate::{
    backend::BackendStorage, cpu_backend::CpuStorage, ops::BinaryOpT, CudaDevice, DType, Error,
};

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

    #[allow(unused_variables, dead_code)]
    pub(crate) fn binary_impl<B: BinaryOpT>(&self, rhs: &Self) -> Result<Self, Error> {
        let lhs_slice = &self.slice;
        let rhs_slice = &rhs.slice;
        match (lhs_slice, rhs_slice) {
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                /*
                                let module = self
                                    .device()
                                    .device
                                    .load_ptx(Ptx::from_src(VECTOR_ADD_KERNEL), "sin", &["sin_kernel"])
                                    .unwrap();
                                let f = self.device().device.get_func("sin", "sin_kernel").unwrap();
                                // Get a reference to the function
                                let a_host = [1.0, 2.0, 3.0, 5.0, 6.0, 7.0];

                                let a_dev = self.device().device.htod_copy(a_host.into()).unwrap();
                                let mut b_dev = a_dev.clone();

                                let n = 3;
                                let cfg = LaunchConfig::for_num_elems(n);
                                unsafe { f.launch(cfg, (&mut b_dev, &a_dev, n as i32)) }.unwrap();

                                let a_host_2 = self.device().device.sync_reclaim(a_dev).unwrap();
                                let b_host = self.device().device.sync_reclaim(b_dev).unwrap();

                                println!("Found {:?}", b_host);
                                println!("Expected {:?}", a_host.map(f32::sin));
                */
                todo!()
            }
            _ => Err(Error::Unknown),
        }
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
