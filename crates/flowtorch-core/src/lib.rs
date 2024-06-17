//! Safe and Performant ML Library for Rust
//!
//! ```rust
//! use flowtorch_core::{DType, Device, Tensor, error::Error};
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::new(&[0.0f32, 1.0, 2.0], &Device::Cpu)?;
//! let b = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
//!
//! let c = (a + b)?;
//! # Ok(())}
//! ```
//!
//! ## Features
//!
//! - Very Similar to PyTorch
//! - Awesome Python support for research workloads
//! - CPU and Cuda support coming soon!
//!
//! ## FAQ
//!
//! - Why FlowTorch?
//!
//! FlowTorch aims to be safe and performant ML library with small codebase for faster pace in development and to eventually to be used
//! in embedded devices and production workloads.
//!

mod array;
mod cpu_backend;
mod device;
mod display;
mod dtype;
pub mod error;
mod formatter;
mod indexer;
mod layout;
mod ops;
mod shape;
mod storage;
mod tensor;
pub use device::{Device, DeviceT};
pub use dtype::DType;
pub use formatter::*;
pub use indexer::*;
pub use shape::Shape;
pub use tensor::*;
mod backend;
mod scalar;

#[cfg(feature = "cuda")]
pub mod cuda_backend;

#[cfg(feature = "cuda")]
pub use cuda_backend as cuda;

#[cfg(not(feature = "cuda"))]
pub mod cuda_failover;

#[cfg(not(feature = "cuda"))]
pub use cuda_failover as cuda;

pub use cuda::CudaDevice;
