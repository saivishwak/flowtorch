use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub struct ArrayError {
    kind: ArrayErrorKind,
}

impl ArrayError {
    pub fn new(kind: ArrayErrorKind) -> Self {
        Self { kind }
    }
}

impl Display for ArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind.as_str())
    }
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum ArrayErrorKind {
    EmptyShape,
    ShapeMismatch,
}

impl ArrayErrorKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::EmptyShape => "Empty Shape",
            Self::ShapeMismatch => "Shape Mismatch",
        }
    }
}

impl Display for ArrayErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
