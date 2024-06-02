use crate::{Error, Tensor};

impl Tensor {
    fn index(&self, indices: &[TensorIdx]) -> Result<Self, Error> {
        if !self.is_layout_contiguous() {
            return Err(Error::Index(String::from(
                "Non Contiguous layout not supported yet!",
            ))); //Not supported as of now
        }

        let mut out = self.clone();
        for (_i, index) in indices.iter().enumerate() {
            out = match index {
                TensorIdx::Select(val) => out.narrow(0, *val, 1)?.squeeze(0)?,
            };
        }

        Ok(out)
    }
}

pub trait IndexOp<T> {
    fn i(&self, index: T) -> Result<Tensor, Error>;
}

impl<T> IndexOp<T> for Tensor
where
    T: Into<TensorIdx>,
{
    fn i(&self, index: T) -> Result<Tensor, Error> {
        self.index(&[index.into()])
    }
}

pub enum TensorIdx {
    Select(usize),
}

impl From<usize> for TensorIdx {
    fn from(value: usize) -> Self {
        TensorIdx::Select(value)
    }
}

macro_rules! index_op_tuple {
    ($($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*> IndexOp<($($t,)*)> for Tensor
        where
            $($t: Into<TensorIdx>,)*
        {
            fn i(&self, ($($t,)*): ($($t,)*)) -> Result<Tensor, Error> {
                self.index(&[$($t.into(),)*])
            }
        }
    };
}

// Generate implementations for tuples of different lengths
index_op_tuple!(T0);
index_op_tuple!(T0, T1);
index_op_tuple!(T0, T1, T2);
index_op_tuple!(T0, T1, T2, T3);
index_op_tuple!(T0, T1, T2, T3, T4);
index_op_tuple!(T0, T1, T2, T3, T4, T5);
