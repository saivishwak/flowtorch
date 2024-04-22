use crate::{
    storage::{self, Storage},
    DType,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn zeros(&self, shape: &[usize], dtype: DType) -> Storage {
        match self {
            Device::Cpu => {
                let elem_count: usize = shape.iter().product();
                let buffer = match dtype {
                    DType::F32 => {
                        let data = vec![0f32; elem_count];
                        data
                    }
                    DType::F64 => {
                        let data = vec![0f64; elem_count];
                        data
                    }
                };
                //let buffer = vec![0; elem_count * dtype.size_in_bytes()];
                Storage::Cpu { dtype, buffer }
            }
        }
    }
}
