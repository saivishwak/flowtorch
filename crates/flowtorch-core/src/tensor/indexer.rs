//use std::{ops::Index, sync::Arc};

//use crate::{layout::Layout, tensor::Tensor_, Error, Tensor};

//TODO - NEED TO FIX
//Indexing should be done with the help of i method which returns a new tensor with a different view

/*
 impl Tensor {
    fn index(&self, idx: &TensorIdx) -> Result<Tensor, Error> {
        let idx_vec = &idx.0;
        if idx_vec.len() < self.dims().len() {
            return Err(Error::Unknown);
        }
        let storage = self.0.storage.clone();
        let binding = self.shape();
        let dims = binding.dims();
        let shape = dims.iter().skip(1).map(|&v| v).collect::<Vec<usize>>();
        let device = self.get_storage_ref().device();
        let tensor_ = Tensor_ {
            storage,
            layout: Layout::contiguous(shape),
            device,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }
}

impl<T> Index<T> for Tensor
where
    T: Into<TensorIdx>,
{
    type Output = Tensor;
    fn index(&self, index: T) -> &Self::Output {
        let tensor_idx: TensorIdx = index.into();
        Box::leak(Box::new(self.index(&tensor_idx).unwrap()))
    }
}

pub struct TensorIdx(Vec<usize>);

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
*/
