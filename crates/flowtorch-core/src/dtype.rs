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
    // Floating-point using single precision (32 bits).
    F32,
    // Floating-point using double precision (64 bits).
    F64,
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::U8 => write!(f, "u8"),
            DType::U32 => write!(f, "u32"),
            DType::I64 => write!(f, "i64"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
        }
    }
}

pub trait WithDType: Sized + Copy + 'static + Debug {
    fn to_cpu_storage(data: &[Self]) -> CpuStorage;
    fn get_dtype() -> DType;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            fn to_cpu_storage(data: &[Self]) -> CpuStorage {
                CpuStorage::$dtype(data.to_vec())
            }

            fn get_dtype() -> DType {
                DType::$dtype
            }
        }
    };
}

with_dtype!(u8, U8);
with_dtype!(u32, U32);
with_dtype!(i64, I64);
with_dtype!(f32, F32);
with_dtype!(f64, F64);
