use std::fmt::Display;

use thiserror::Error;

use crate::{cpu_backend::CpuStorageError, ndarray::NDArrayError, DType};

#[derive(Error, Debug)]
pub enum Error {
    #[error("Tensor initialization failed with Dtype {0:?} due to {1}")]
    TensorInit(Option<DType>, String),
    #[error("{0}")]
    Shape(ShapeError),
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
    #[error("NDArray")]
    NDArray {
        #[from]
        source: NDArrayError,
    },
    #[error("Index error")]
    Index(String),
    #[error("unknown error")]
    Unknown,
}

#[derive(Error, Debug)]
pub enum ShapeError {
    EmptyShape,
    ShapeMismatch,
    ReshapeError(String),
    Narrow(String),
    CustomError,
}

impl Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyShape => write!(f, "Empty Shape"),
            Self::ShapeMismatch => write!(f, "Shape Mismatch"),
            Self::CustomError => write!(f, "{}", self),
            Self::ReshapeError(msg) => write!(f, "{}", msg),
            Self::Narrow(msg) => write!(f, "{}", msg),
        }
    }
}

#[derive(Error, Debug)]
pub struct CustomError<'a> {
    msg: &'a str,
}

impl<'a> CustomError<'a> {
    pub fn new(str: &'a str) -> Self {
        Self { msg: str }
    }
}

impl<'a> Display for CustomError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

#[derive(Error, Debug)]
pub enum StorageErrorKind {
    CpuStorage(CpuStorageError),
}

impl StorageErrorKind {
    pub fn as_string(&self) -> String {
        match self {
            Self::CpuStorage(cpu_storage_error) => format!("{}", cpu_storage_error),
        }
    }
}

impl Display for StorageErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

#[derive(Error, Debug)]
pub struct StorageError {
    #[from]
    source: StorageErrorKind,
}

impl Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source.as_string())
    }
}

#[derive(Error, Debug)]
pub enum DeviceErrorKind {
    ZerosFail,
    OnesFail,
    FromArrayFailure,
}

impl DeviceErrorKind {
    pub fn as_string(&self) -> String {
        match self {
            Self::ZerosFail => format!("Zeros initialization failed!"),
            Self::FromArrayFailure => format!("From Array Failed"),
            Self::OnesFail => format!("Ones initialization failed!"),
        }
    }
}

impl Display for DeviceErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

#[derive(Error, Debug)]
pub struct DeviceError {
    kind: DeviceErrorKind,
}

impl DeviceError {
    pub fn new(kind: DeviceErrorKind) -> Self {
        Self { kind }
    }
}

impl Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind.as_string())
    }
}
