#![allow(dead_code, unused_imports, unused_variables)]
use crate::{layout::Layout, ops::Op, Tensor};
use std::{f32::INFINITY, fmt::Display};

// Should adhere to https://github.com/pytorch/pytorch/blob/7b419e8513a024e172eae767e24ec1b849976b13/torch/_tensor_str.py

#[derive(Debug, Clone)]
pub enum PrintProfiles {
    Default,
    Short,
    Full,
}

#[derive(Debug, Clone)]
pub struct PrintOptions {
    precision: usize,
    threshold: f32,
    edgeitems: usize,
    linewidth: usize,
    profile: PrintProfiles,
}

impl PrintOptions {
    pub fn new() -> Self {
        Self {
            precision: 4,
            threshold: 1000.0,
            edgeitems: 3,
            linewidth: 80,
            profile: PrintProfiles::Default,
        }
    }

    pub fn set_precision(&mut self, precision: usize) -> &Self {
        self.precision = precision;
        self
    }

    pub fn set_threshold(&mut self, threshold: f32) -> &Self {
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
                self.threshold = 1000.0;
                self.edgeitems = 3;
                self.linewidth = 80;
            }
            PrintProfiles::Short => {
                self.precision = 2;
                self.threshold = 1000.0;
                self.edgeitems = 2;
                self.linewidth = 80;
            }
            PrintProfiles::Full => {
                self.precision = 4;
                self.threshold = INFINITY;
                self.edgeitems = 3;
                self.linewidth = 80;
            }
        };
        self
    }
}

impl Tensor {
    pub fn fmt(&self, options: Option<PrintOptions>) -> Result<String, String> {
        let print_options = match options {
            Some(opts) => opts,
            None => PrintOptions::new(),
        };
        let layout = self.layout().clone();
        let storage = self.get_storage_ref();
        let binding = storage.cpu_get_raw();
        let cpu_storage_data = binding.as_ref();

        if !self.is_layout_contiguous() {
            return Err(String::from("Non Contigous layout not supported"));
        }
        let formatted_string = match cpu_storage_data {
            crate::cpu_backend::CpuStorage::U8(data) => {
                fmt_tensor_as_string(&data, layout, print_options)
            }
            crate::cpu_backend::CpuStorage::U32(data) => {
                fmt_tensor_as_string(&data, layout, print_options)
            }
            crate::cpu_backend::CpuStorage::I64(data) => {
                fmt_tensor_as_string(&data, layout, print_options)
            }
            crate::cpu_backend::CpuStorage::F32(data) => {
                fmt_tensor_as_string(&data, layout, print_options)
            }
            crate::cpu_backend::CpuStorage::F64(data) => {
                fmt_tensor_as_string(&data, layout, print_options)
            }
        };
        Ok(formatted_string)
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let formatted_string = self.fmt(None);
        match formatted_string {
            Ok(string) => {
                return write!(
                    f,
                    "Tensor({}, dtype; {}, device; {})",
                    string,
                    self.dtype(),
                    self.device()
                );
            }
            Err(e) => {
                return write!(f, "{}", e);
            }
        }
    }
}

//TODO - Need to fix usage of printoptions
pub fn fmt_tensor_as_string<T: std::fmt::Display>(
    data: &Vec<T>,
    layout: Layout,
    options: PrintOptions,
) -> String {
    let mut result = String::new();
    let shape = layout.shape();
    let strides = layout.stride();
    let dims = shape.dims();
    let initial_offset = layout.start_offset();

    if data.len() == 0 {
        return result;
    }

    //Scalar value
    if dims.len() == 0 {
        result.push_str(format!("{:.1$}", data[initial_offset], 2).as_str());
        return result;
    }

    for _ in 0..dims.len() {
        result.push_str("[");
    }

    let mut sub_array_end_detected: bool;

    let num_elms = dims.iter().map(|f| *f).product();
    let mut i = initial_offset;
    let mut count: usize = 0;

    while i < data.len() && count < num_elms {
        result.push_str(format!("{:.1$}", data[i], options.clone().precision).as_str());

        // In each dimension should we end the subarray with ']'
        let dim_end_len = dims
            .iter()
            .zip(&strides)
            .filter(|&(dim, stride)| (i + 1) % (*dim * (*stride)) == 0)
            .count();

        if dim_end_len > 0 {
            for _ in 0..dim_end_len {
                result.push_str("]"); //Put in the ']' for all ending places like [1]
            }
            sub_array_end_detected = true;
        } else {
            sub_array_end_detected = false;
        }

        if i != data.len() - 1 {
            result.push_str(", "); // If the index is not last add ','

            if sub_array_end_detected {
                for _k in 0..dim_end_len {
                    result.push_str("[");
                }
            }
        }
        i += 1;
        count += 1;
    }

    return result;
}
