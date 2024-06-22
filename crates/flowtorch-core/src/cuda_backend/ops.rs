use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::{
    dtype::WithDType,
    error::StorageError,
    ops::{BinaryOpT, UnaryOpT},
    CudaDevice, Layout,
};

use super::{error::CudaStorageError, storage::CudaStorageSlice, utils, CudaStorage};
use flowcuda_kernels as kernels;

type S = CudaStorageSlice;

pub trait Pair1Runner {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
        layout: &Layout,
    ) -> Result<CudaSlice<T>, StorageError>;
    fn run_op(
        &self,
        device: CudaDevice,
        lhs: &CudaStorage,
        layout: &Layout,
    ) -> Result<S, StorageError>;
}

pub trait Pair2Runner {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
        rhs: &CudaSlice<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<CudaSlice<T>, StorageError>;
    fn run_op(
        &self,
        device: CudaDevice,
        lhs: &CudaStorage,
        rhs: &CudaStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<CudaStorageSlice, StorageError>;
}
//lhs_is_contiguous, size_t *lhs_layout, size_t lhs_num_dims, bool rhs_is_contiguous, size_t *rhs_layout, size_t rhs_num_dims)
impl<B: BinaryOpT> Pair2Runner for B {
    fn run<T: WithDType + cudarc::driver::DeviceRepr>(
        &self,
        device: CudaDevice,
        lhs: &CudaSlice<T>,
        rhs: &CudaSlice<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<CudaSlice<T>, StorageError> {
        let numel = lhs.len();
        let func = match device
            .get_and_load_kernal_func(&utils::get_kernel_name::<T>(B::KERNEL), kernels::BINARY)
        {
            Ok(f) => f,
            Err(e) => return Err(CudaStorageError::DeviceError { source: e }.into()),
        };
        let data = device.alloc::<T>(numel).unwrap();
        let lhs_layout_data = device
            .device
            .htod_copy([lhs_layout.shape().dims(), lhs_layout.stride().as_slice()].concat())
            .unwrap(); //It is okay to panic here if we are not able to alloc data
        let rhs_layout_data = device
            .device
            .htod_copy([rhs_layout.shape().dims(), rhs_layout.stride().as_slice()].concat())
            .unwrap(); //It is okay to panic here if we are not able to alloc data
        let launch_config: LaunchConfig = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,
            lhs,
            rhs,
            &data,
            lhs_layout.is_contiguous(),
            &lhs_layout_data,
            lhs_layout.shape().dims().len(),
            rhs_layout.is_contiguous(),
            &rhs_layout_data,
            rhs_layout.shape().dims().len(),
        );
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
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<S, StorageError> {
        let lhs_slice = &lhs.slice;
        let rhs_slice = &rhs.slice;
        match (lhs_slice, rhs_slice) {
            (S::F32(lhs), S::F32(rhs)) => {
                Ok(S::F32(self.run(device, lhs, rhs, lhs_layout, rhs_layout)?))
            }
            (S::F64(lhs), S::F64(rhs)) => {
                Ok(S::F64(self.run(device, lhs, rhs, lhs_layout, rhs_layout)?))
            }
            (S::I32(lhs), S::I32(rhs)) => {
                Ok(S::I32(self.run(device, lhs, rhs, lhs_layout, rhs_layout)?))
            }
            (S::I64(lhs), S::I64(rhs)) => {
                Ok(S::I64(self.run(device, lhs, rhs, lhs_layout, rhs_layout)?))
            }
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
        layout: &Layout,
    ) -> Result<CudaSlice<T>, StorageError> {
        let numel = lhs.len();
        let num_dims = layout.shape().dims().len();
        let func = match device
            .get_and_load_kernal_func(&utils::get_kernel_name::<T>(U::KERNEL), kernels::UNARY)
        {
            Ok(f) => f,
            Err(e) => return Err(CudaStorageError::DeviceError { source: e }.into()),
        };
        let data = device.alloc::<T>(numel).unwrap();
        let launch_config = LaunchConfig::for_num_elems(numel as u32);
        let layout_data = device
            .device
            .htod_copy([layout.shape().dims(), layout.stride().as_slice()].concat())
            .unwrap(); //It is okay to panic here if we are not able to alloc data
        let params = (
            numel,
            lhs,
            &data,
            layout.is_contiguous(),
            &layout_data,
            num_dims,
        );
        match unsafe { func.launch(launch_config, params) } {
            Ok(_) => Ok(data),
            Err(e) => Err(CudaStorageError::OpRunner(e.to_string()).into()),
        }
    }
    fn run_op(
        &self,
        device: CudaDevice,
        lhs: &CudaStorage,
        layout: &Layout,
    ) -> Result<S, StorageError> {
        let slice = &lhs.slice;
        match slice {
            S::F32(lhs) => Ok(S::F32(self.run(device, lhs, layout)?)),
            S::F64(lhs) => Ok(S::F64(self.run(device, lhs, layout)?)),
            S::I32(lhs) => Ok(S::I32(self.run(device, lhs, layout)?)),
            S::I64(lhs) => Ok(S::I64(self.run(device, lhs, layout)?)),
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
