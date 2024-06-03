use crate::{dtype::WithDType, Tensor};
use std::vec;

// Should adhere to https://github.com/pytorch/pytorch/blob/7b419e8513a024e172eae767e24ec1b849976b13/torch/_tensor_str.py

#[derive(Debug, Clone, Copy)]
pub enum PrintProfiles {
    Default,
    Short,
    Full,
}

//TODO Use linewidth to add '\n' then we in line with Pytorch
pub struct Formatter {
    _max_width: usize,
    tensor: Tensor,
    options: PrintOptions,
}

impl Formatter {
    pub fn new(tensor: Tensor, options: PrintOptions) -> Self {
        Self {
            _max_width: 1,
            tensor,
            options,
        }
    }

    pub fn fmt(&self) -> Result<String, String> {
        let storage = self.tensor.storage();
        let cpu_storage = *storage.cpu_get_raw();
        let layout = self.tensor.layout();
        let shape = layout.shape();
        let dims = shape.dims();
        let stride = layout.stride();
        let start_offset = layout.start_offset();

        if !self.tensor.is_layout_contiguous() {
            return Err(String::from("Non Contiguous layout not supported!"));
        }
        let formatted_string = match cpu_storage {
            crate::cpu_backend::CpuStorage::U8(data) => {
                self.fmt_tensor_as_string(data, dims, &stride, start_offset)
            }
            crate::cpu_backend::CpuStorage::U32(data) => {
                self.fmt_tensor_as_string(data, dims, &stride, start_offset)
            }
            crate::cpu_backend::CpuStorage::I64(data) => {
                self.fmt_tensor_as_string(data, dims, &stride, start_offset)
            }
            crate::cpu_backend::CpuStorage::I32(data) => {
                self.fmt_tensor_as_string(data, dims, &stride, start_offset)
            }
            crate::cpu_backend::CpuStorage::F32(data) => {
                self.fmt_tensor_as_string(data, dims, &stride, start_offset)
            }
            crate::cpu_backend::CpuStorage::F64(data) => {
                self.fmt_tensor_as_string(data, dims, &stride, start_offset)
            }
        };
        Ok(formatted_string)
    }

    fn fmt_scalar_str<T: WithDType>(&self, scalar: T) -> String {
        format!("{:.1$}", scalar, self.options.precision)
    }

    fn fmt_vector_str<T: WithDType>(
        &self,
        data: &[T],
        shape: &[usize],
        stride: &[usize],
        indices: &mut Vec<usize>,
        depth: usize,
        start: usize,
        result: &mut Vec<String>,
        edgeitems: usize,
    ) {
        if depth == shape.len() {
            let linear_index = indices
                .iter()
                .zip(stride)
                .map(|(&i, &s)| i * s)
                .sum::<usize>();
            result.push(self.fmt_scalar_str(data[start + linear_index]));
            return;
        }

        if shape[depth] > edgeitems * 2 && shape[depth] > self.options.threshold {
            // Summarize by showing only edgeitems at the beginning and end
            let mut row = Vec::new();
            for i in 0..edgeitems {
                indices[depth] = i;
                self.fmt_vector_str(
                    data,
                    shape,
                    stride,
                    indices,
                    depth + 1,
                    start,
                    &mut row,
                    edgeitems,
                );
            }
            row.push(String::from(" ... "));
            for i in (shape[depth] - edgeitems)..shape[depth] {
                indices[depth] = i;
                self.fmt_vector_str(
                    data,
                    shape,
                    stride,
                    indices,
                    depth + 1,
                    start,
                    &mut row,
                    edgeitems,
                );
            }
            result.push(format!("[{}]", row.join(", ")));
        } else {
            // Full representation
            let mut row = Vec::new();
            for i in 0..shape[depth] {
                indices[depth] = i;
                self.fmt_vector_str(
                    data,
                    shape,
                    stride,
                    indices,
                    depth + 1,
                    start,
                    &mut row,
                    edgeitems,
                );
            }
            result.push(format!("[{}]", row.join(", ")));
        }
    }

    fn fmt_tensor_as_string<T>(
        &self,
        data: &[T],
        dims: &[usize],
        stride: &[usize],
        initial_offset: usize,
    ) -> String
    where
        T: WithDType,
    {
        if data.len() == 0 {
            return String::new();
        }
        //Scalar
        if dims.len() == 0 {
            return self.fmt_scalar_str(data[initial_offset]);
        }

        let mut result: Vec<String> = vec![];
        let mut indices = vec![0; stride.len()];

        self.fmt_vector_str(
            data,
            dims,
            stride,
            &mut indices,
            0,
            initial_offset,
            &mut result,
            self.options.edgeitems,
        );

        format!("{}", result.join(", "))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrintOptions {
    precision: usize,
    threshold: usize,
    edgeitems: usize,
    linewidth: usize,
    profile: PrintProfiles,
}

impl PrintOptions {
    pub fn new() -> Self {
        Self {
            precision: 4,
            threshold: 1000,
            edgeitems: 3,
            linewidth: 80,
            profile: PrintProfiles::Default,
        }
    }

    pub fn set_precision(&mut self, precision: usize) -> &Self {
        self.precision = precision;
        self
    }

    pub fn set_threshold(&mut self, threshold: usize) -> &Self {
        self.threshold = threshold;
        self
    }

    pub fn set_edgeitems(&mut self, edgeitems: usize) -> &Self {
        self.edgeitems = edgeitems;
        self
    }

    pub fn set_linewidth(&mut self, linewidth: usize) -> &Self {
        self.linewidth = linewidth;
        self
    }

    pub fn set_profile(&mut self, profile: PrintProfiles) -> &Self {
        match profile {
            PrintProfiles::Default => {
                self.precision = 4;
                self.threshold = 1000;
                self.edgeitems = 3;
                self.linewidth = 80;
            }
            PrintProfiles::Short => {
                self.precision = 2;
                self.threshold = 1000;
                self.edgeitems = 2;
                self.linewidth = 80;
            }
            PrintProfiles::Full => {
                self.precision = 4;
                self.threshold = std::usize::MAX;
                self.edgeitems = 3;
                self.linewidth = 80;
            }
        };
        self
    }

    pub fn profile(&self) -> PrintProfiles {
        self.profile
    }
}
