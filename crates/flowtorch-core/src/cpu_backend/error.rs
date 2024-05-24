use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug)]
pub struct CpuStorageError {
    kind: CpuStorageErrorKind,
}

impl CpuStorageError {
    pub fn new(kind: CpuStorageErrorKind) -> Self {
        Self { kind }
    }

    pub fn as_str(&self) -> String {
        self.kind.as_string()
    }
}

impl Display for CpuStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind.as_string())
    }
}

#[derive(Debug)]
pub enum CpuStorageErrorKind {
    Custom(String),
    MismatchDtype,
    ContiguousElementDtypeMismatch,
}

impl CpuStorageErrorKind {
    pub fn as_string(&self) -> String {
        match self {
            Self::Custom(msg) => format!("Custom CpuStorageError {}", msg.clone()),
            Self::MismatchDtype => String::from("Mismatch Data Type"),
            Self::ContiguousElementDtypeMismatch => {
                format!("Some of the elements have different DType")
            }
        }
    }
}
