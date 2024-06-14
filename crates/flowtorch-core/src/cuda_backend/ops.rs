use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::{
    dtype::WithDType,
    ops::{BinaryOpT, UnaryOpT},
    CudaDevice, Error,
};

use super::{storage::CudaStorageSlice, utils, CudaStorage};
use flowcuda_kernels as kernels;

pub trait BinrayRunOp {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
        rhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, Error>;
    fn run_op(
        &self,
        device: CudaDevice,
        lhs: &CudaStorage,
        rhs: &CudaStorage,
    ) -> Result<CudaStorageSlice, Error>;
}

pub trait UnaryRunOp {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, Error>;
    fn run_op(&self, device: CudaDevice, lhs: &CudaStorage) -> Result<CudaStorageSlice, Error>;
}

impl<B: BinaryOpT> BinrayRunOp for B {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
        rhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, Error> {
        let numel = lhs.len();
        let func = device
            .get_and_load_kernal_func(&utils::get_kernel_name::<T>(B::KERNEL), kernels::BINARY)?;
        let data = device.alloc::<T>(numel).unwrap();
        let launch_config = LaunchConfig::for_num_elems(numel as u32);
        let params = (numel, lhs, rhs, &data);
        match unsafe { func.launch(launch_config, params) } {
            Ok(_) => {
                return Ok(data);
            }
            Err(_) => {
                return Err(Error::Unknown);
            }
        }
    }
    fn run_op(
        &self,
        device: CudaDevice,
        lhs: &CudaStorage,
        rhs: &CudaStorage,
    ) -> Result<CudaStorageSlice, Error> {
        let lhs_slice = &lhs.slice;
        let rhs_slice = &rhs.slice;
        match (lhs_slice, rhs_slice) {
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let s = self.run(device, lhs, rhs)?;
                Ok(CudaStorageSlice::F32(s))
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                let s = self.run(device, lhs, rhs)?;
                Ok(CudaStorageSlice::F64(s))
            }
            (CudaStorageSlice::I32(lhs), CudaStorageSlice::I32(rhs)) => {
                let s = self.run(device, lhs, rhs)?;
                Ok(CudaStorageSlice::I32(s))
            }
            (CudaStorageSlice::I64(lhs), CudaStorageSlice::I64(rhs)) => {
                let s = self.run(device, lhs, rhs)?;
                Ok(CudaStorageSlice::I64(s))
            }
            _ => {
                //Not supported right now for U8 and U32
                Err(Error::Unknown)
            }
        }
    }
}

impl<U: UnaryOpT> UnaryRunOp for U {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
    ) -> Result<CudaSlice<T>, Error> {
        let numel = lhs.len();
        println!("TESTS {}", &utils::get_kernel_name::<T>(U::KERNEL));
        let func = device
            .get_and_load_kernal_func(&utils::get_kernel_name::<T>(U::KERNEL), kernels::UNARY)?;
        let data = device.alloc::<T>(numel).unwrap();
        let launch_config = LaunchConfig::for_num_elems(numel as u32);
        let params = (numel, lhs, &data);
        match unsafe { func.launch(launch_config, params) } {
            Ok(_) => {
                return Ok(data);
            }
            Err(_) => {
                return Err(Error::Unknown);
            }
        }
    }
    fn run_op(&self, device: CudaDevice, lhs: &CudaStorage) -> Result<CudaStorageSlice, Error> {
        let slice = &lhs.slice;
        match slice {
            CudaStorageSlice::F32(lhs) => {
                let s = self.run(device, lhs)?;
                Ok(CudaStorageSlice::F32(s))
            }
            CudaStorageSlice::F64(lhs) => {
                let s = self.run(device, lhs)?;
                Ok(CudaStorageSlice::F64(s))
            }
            CudaStorageSlice::I32(lhs) => {
                let s = self.run(device, lhs)?;
                Ok(CudaStorageSlice::I32(s))
            }
            CudaStorageSlice::I64(lhs) => {
                let s = self.run(device, lhs)?;
                Ok(CudaStorageSlice::I64(s))
            }
            _ => {
                //Not supported right now for U8 and U32
                Err(Error::Unknown)
            }
        }
    }
}
