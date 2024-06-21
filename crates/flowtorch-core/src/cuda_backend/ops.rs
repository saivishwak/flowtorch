use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::{
    dtype::WithDType,
    error::StorageError,
    ops::{BinaryOpT, UnaryOpT},
    CudaDevice,
};

use super::{error::CudaStorageError, storage::CudaStorageSlice, utils, CudaStorage};
use flowcuda_kernels as kernels;

type S = CudaStorageSlice;

pub trait Pair1Runner {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, StorageError>;
    fn run_op(&self, device: CudaDevice, lhs: &CudaStorage) -> Result<S, StorageError>;
}

pub trait Pair2Runner {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
        rhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, StorageError>;
    fn run_op(
        &self,
        device: CudaDevice,
        lhs: &CudaStorage,
        rhs: &CudaStorage,
    ) -> Result<CudaStorageSlice, StorageError>;
}

impl<B: BinaryOpT> Pair2Runner for B {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
        rhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, StorageError> {
        let numel = lhs.len();
        let func = match device
            .get_and_load_kernal_func(&utils::get_kernel_name::<T>(B::KERNEL), kernels::BINARY)
        {
            Ok(f) => f,
            Err(e) => return Err(CudaStorageError::DeviceError { source: e }.into()),
        };
        let data = device.alloc::<T>(numel).unwrap();
        let launch_config: LaunchConfig = LaunchConfig::for_num_elems(numel as u32);
        let params = (numel, lhs, rhs, &data);
        match unsafe { func.launch(launch_config, params) } {
            Ok(_) => Ok(data),
            Err(e) => Err(CudaStorageError::OpRunner(e.to_string()).into()),
        }
    }
    fn run_op(
        &self,
        device: CudaDevice,
        lhs: &CudaStorage,
        rhs: &CudaStorage,
    ) -> Result<S, StorageError> {
        let lhs_slice = &lhs.slice;
        let rhs_slice = &rhs.slice;
        match (lhs_slice, rhs_slice) {
            (S::F32(lhs), S::F32(rhs)) => Ok(S::F32(self.run(device, lhs, rhs)?)),
            (S::F64(lhs), S::F64(rhs)) => Ok(S::F64(self.run(device, lhs, rhs)?)),
            (S::I32(lhs), S::I32(rhs)) => Ok(S::I32(self.run(device, lhs, rhs)?)),
            (S::I64(lhs), S::I64(rhs)) => Ok(S::I64(self.run(device, lhs, rhs)?)),
            _ => {
                //Not supported right now for U8 and U32
                Err(
                    CudaStorageError::OpRunner("U8 and U32 are not yet supported".to_string())
                        .into(),
                )
            }
        }
    }
}

impl<U: UnaryOpT> Pair1Runner for U {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, StorageError> {
        let numel = lhs.len();
        let func = match device
            .get_and_load_kernal_func(&utils::get_kernel_name::<T>(U::KERNEL), kernels::UNARY)
        {
            Ok(f) => f,
            Err(e) => return Err(CudaStorageError::DeviceError { source: e }.into()),
        };
        let data = device.alloc::<T>(numel).unwrap();
        let launch_config = LaunchConfig::for_num_elems(numel as u32);
        let params = (numel, lhs, &data);
        match unsafe { func.launch(launch_config, params) } {
            Ok(_) => Ok(data),
            Err(e) => Err(CudaStorageError::OpRunner(e.to_string()).into()),
        }
    }
    fn run_op(&self, device: CudaDevice, lhs: &CudaStorage) -> Result<S, StorageError> {
        let slice = &lhs.slice;
        match slice {
            S::F32(lhs) => Ok(S::F32(self.run(device, lhs)?)),
            S::F64(lhs) => Ok(S::F64(self.run(device, lhs)?)),
            S::I32(lhs) => Ok(S::I32(self.run(device, lhs)?)),
            S::I64(lhs) => Ok(S::I64(self.run(device, lhs)?)),
            _ => {
                //Not supported right now for U8 and U32
                Err(
                    CudaStorageError::OpRunner("U8 and U32 are not yet supported".to_string())
                        .into(),
                )
            }
        }
    }
}
