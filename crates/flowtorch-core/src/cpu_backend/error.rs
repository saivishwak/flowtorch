use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CpuStorageError {
    Custom(String),
    MismatchDtype,
    ContiguousElementDtypeMismatch,
    EmptyArray,
}

impl CpuStorageError {
    pub fn as_string(&self) -> String {
        match self {
            Self::Custom(msg) => format!("Custom CpuStorageError {}", msg.clone()),
            Self::MismatchDtype => String::from("Mismatch Data Type"),
            Self::ContiguousElementDtypeMismatch => {
                "Some of the elements have different DType".to_string()
            }
            Self::EmptyArray => "Array should not be empty".to_string(),
        }
    }
}

impl Display for CpuStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CpuDeviceError {
    Custom(String),
    MismatchDtype,
}

impl CpuDeviceError {
    pub fn as_string(&self) -> String {
        match self {
            Self::Custom(msg) => format!("Custom CpuStorageError {}", msg),
            Self::MismatchDtype => String::from("Mismatch Data Type"),
        }
    }
}

impl Display for CpuDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}
