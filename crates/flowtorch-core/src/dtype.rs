use std::fmt::{Debug, Display};

use crate::cpu_backend::CpuStorage;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    // Unsigned 8 bits integer.
    U8,
    // Unsigned 32 bits integer.
    U32,
    // Signed 64 bits integer.
    I64,
    // Signed 32 bits integer.
    I32,
    // Floating-point using single precision (32 bits).
    F32,
    // Floating-point using double precision (64 bits).
    F64,
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U8 => write!(f, "u8"),
            Self::U32 => write!(f, "u32"),
            Self::I64 => write!(f, "i64"),
            Self::I32 => write!(f, "i32"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
        }
    }
}

pub trait WithDType: Sized + Copy + 'static + Debug + Display {
    fn to_cpu_storage(data: &[Self]) -> CpuStorage;
    fn dtype() -> DType;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            fn to_cpu_storage(data: &[Self]) -> CpuStorage {
                CpuStorage::$dtype(data.to_vec())
            }

            fn dtype() -> DType {
                DType::$dtype
            }
        }
    };
}

with_dtype!(u8, U8);
with_dtype!(u32, U32);
with_dtype!(i64, I64);
with_dtype!(i32, I32);
with_dtype!(f32, F32);
with_dtype!(f64, F64);
