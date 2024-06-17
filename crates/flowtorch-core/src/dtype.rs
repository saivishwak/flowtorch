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

impl DType {
    pub fn as_str(&self) -> String {
        match self {
            Self::U8 => String::from("u8"),
            Self::U32 => String::from("u32"),
            Self::I64 => String::from("i64"),
            Self::I32 => String::from("i32"),
            Self::F32 => String::from("f32"),
            Self::F64 => String::from("f64"),
        }
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = self.as_str();
        write!(f, "{}", str)
    }
}

pub trait WithDType: Sized + Copy + 'static + Debug + Display {
    fn to_cpu_storage(data: &[Self]) -> CpuStorage;
    fn dtype() -> DType;
    //Casting Methods
    fn to_f32(&self) -> f32;
    fn to_f64(&self) -> f64;
    fn to_i32(&self) -> i32;
    fn to_i64(&self) -> i64;
    fn to_u8(&self) -> u8;
    fn to_u32(&self) -> u32;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            fn to_cpu_storage(data: &[Self]) -> CpuStorage {
                //Optimization to limit vector capacity to 1 as these are scalars
                let mut vec: Vec<$ty> = Vec::with_capacity(1);
                vec.extend_from_slice(data);
                CpuStorage::$dtype(vec)
            }

            fn dtype() -> DType {
                DType::$dtype
            }

            fn to_f32(&self) -> f32 {
                *self as f32
            }

            fn to_f64(&self) -> f64 {
                *self as f64
            }

            fn to_i64(&self) -> i64 {
                *self as i64
            }

            fn to_i32(&self) -> i32 {
                *self as i32
            }

            fn to_u8(&self) -> u8 {
                *self as u8
            }

            fn to_u32(&self) -> u32 {
                *self as u32
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
