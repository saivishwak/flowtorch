use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug)]
pub struct NDArrayError {
    kind: NDArrayErrorKind,
}

impl NDArrayError {
    pub fn new(kind: NDArrayErrorKind) -> Self {
        Self { kind }
    }
}

impl Display for NDArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind.as_str())
    }
}

#[derive(Debug)]
pub enum NDArrayErrorKind {
    EmptyShape,
    ShapeMismatch,
}

impl NDArrayErrorKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::EmptyShape => "Empty Shape",
            Self::ShapeMismatch => "Shape Mismatch",
        }
    }
}

impl Display for NDArrayErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
