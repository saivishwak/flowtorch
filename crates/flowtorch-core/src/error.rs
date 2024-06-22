use thiserror::Error;

use crate::{
    array::ArrayError,
    cpu_backend::{CpuDeviceError, CpuStorageError},
    cuda::{CudaDeviceError, CudaStorageError},
    DType,
};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum LayoutError {
    #[error("Empty Shape")]
    EmptyShape,
    #[error("Shape Mismatch")]
    ShapeMismatch,
    #[error("ReshapeError: {0}")]
    ReshapeError(String),
    #[error("Narrow Error: {0}")]
    Narrow(String),
    #[error("CustomError: {0}")]
    CustomError(String),
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum StorageError {
    #[error("CPU Storage Error: {}", .source)]
    Cpu {
        #[from]
        source: CpuStorageError,
    },
    #[error("CUDA Storage Error: {}", .source)]
    Cuda {
        #[from]
        source: CudaStorageError,
    },
    #[error("From Device Error {}", .soruce)]
    Deivce {
        #[from]
        soruce: DeviceError,
    },
    #[error("Deivce Mismatch")]
    DeviceMismatch,
    #[error("DataType Mismatch for {0} and {0}")]
    DataTypeMismatch(&'static str, &'static str),
    #[error("unknown error")]
    Unknown, //Should be use only when we cannot determine the error, which will mostly likely not occur
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum DeviceError {
    #[error("CPU Device Error: {}", .source)]
    Cpu {
        #[from]
        source: CpuDeviceError,
    },
    #[error("CUDA Device Error: {}", .source)]
    Cuda {
        #[from]
        source: CudaDeviceError,
    },
    #[error("Zeros initialization failed")]
    ZerosFail,
    #[error("Ones initialization failed")]
    OnesFail,
    #[error("Zeros initialization failed")]
    InitFail,
    #[error("Zeros initialization failed")]
    FromArrayFailure,
    #[error("Zeros initialization failed")]
    DeviceMismatch,
    #[error("Zeros initialization failed")]
    CopyFail,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum OpError {
    #[error("Binary Op Error: {0}")]
    Binary(String),
    #[error("Unary Op Error: {0}")]
    Unary(String),
    #[error("CustomError: {0}")]
    CustomError(String),
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum Error {
    #[error("Tensor initialization failed with Dtype {0:?} due to {1}")]
    TensorInit(Option<DType>, String),
    #[error("Layout Error : {}",.source)]
    Layout {
        #[from]
        source: LayoutError,
    },
    #[error("Error in Device")]
    Device {
        #[from]
        source: DeviceError,
    },
    #[error("Storage Error {}", .source)]
    Storage {
        #[from]
        source: StorageError,
    },
    #[error("Array")]
    Array {
        #[from]
        source: ArrayError,
    },
    #[error("Indexing error: {0}")]
    Index(String),
    #[error("Op Error: {}", .source)]
    Op {
        #[from]
        source: OpError,
    },
    #[error("{0}")]
    Unimplemented(&'static str),
    #[error("unknown error")]
    Unknown, //Should be use only when we cannot determine the error, which will mostly likely not occur
}
