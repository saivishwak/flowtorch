use crate::{layout::Layout, tensor::Tensor_, Error, Tensor};
use std::sync::Arc;

impl Tensor {
    pub fn i<T: Into<TensorIdx>>(&self, idx: T) -> Result<Tensor, Error> {
        if !self.is_layout_contiguous() {
            return Err(Error::Index(String::from(
                "Non Contiguous layout not supported yet!",
            ))); //Not supported as of now
        }
        //Initial offfset
        let mut offset = self.offset();
        let idx: TensorIdx = idx.into(); //Safe to call into directly
        let idx_vec = idx.0;
        let dims = self.dims();
        let strides = self.stride();

        if idx_vec.len() == 0 {
            return Err(Error::Index(String::from("Index value cannot be empty.")));
        }
        if idx_vec.len() > dims.len() {
            return Err(Error::Index(String::from(
                "Dimentionality mismatch of index and Tensor Dim.",
            )));
        }
        //Check overflow in each dim
        for i in 0..idx_vec.len() {
            if idx_vec[i] >= dims[i] {
                return Err(Error::Index(format!("Overflow occured at Dimension {}", i)));
            }
        }

        let storage = self.get_storage_clone();

        let shape = dims
            .iter()
            .skip(idx_vec.len())
            .map(|&v| v)
            .collect::<Vec<usize>>();

        //Calculate offset
        for i in 0..idx_vec.len() {
            offset += strides[i] * idx_vec[i];
        }

        let device = self.get_storage_ref().device();
        let tensor_ = Tensor_ {
            storage,
            layout: Layout::contiguous_with_offset(shape, offset),
            device,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }
}

pub struct TensorIdx(Vec<usize>);

impl From<Vec<usize>> for TensorIdx {
    fn from(value: Vec<usize>) -> Self {
        TensorIdx(Vec::from(value))
    }
}

//TODO Add more dimension support
impl From<(usize,)> for TensorIdx {
    fn from(value: (usize,)) -> Self {
        let dim1 = value.0;
        TensorIdx(vec![dim1])
    }
}

impl From<(usize, usize)> for TensorIdx {
    fn from(value: (usize, usize)) -> Self {
        let dim1 = value.0;
        let dim2 = value.1;
        TensorIdx(vec![dim1, dim2])
    }
}
