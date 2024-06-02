use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::{Error, Tensor};

impl Tensor {
    fn index(&self, indices: &[TensorIdx]) -> Result<Self, Error> {
        if !self.is_layout_contiguous() {
            return Err(Error::Index(String::from(
                "Non Contiguous layout not supported yet!",
            ))); //Not supported as of now
        }

        let shape = self.shape();
        let dims = shape.dims();
        let mut out = self.clone();
        let mut current_dim = 0;
        for (i, index) in indices.iter().enumerate() {
            out = match index {
                TensorIdx::Select(val) => out.narrow(0, *val, 1)?.squeeze(0)?,
                TensorIdx::Narrow(left_bound, right_bound) => {
                    let start = match left_bound {
                        Bound::Included(n) => *n,
                        Bound::Excluded(n) => *n + 1,
                        Bound::Unbounded => 0,
                    };
                    let stop = match right_bound {
                        Bound::Included(n) => *n + 1,
                        Bound::Excluded(n) => *n,
                        Bound::Unbounded => dims[i],
                    };
                    let x = out.narrow(current_dim, start, stop.saturating_sub(start))?;
                    current_dim += 1;
                    x
                }
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
    Narrow(Bound<usize>, Bound<usize>),
}

impl From<usize> for TensorIdx {
    fn from(value: usize) -> Self {
        TensorIdx::Select(value)
    }
}

trait RB: RangeBounds<usize> {}
impl RB for Range<usize> {}
impl RB for RangeFrom<usize> {}
impl RB for RangeFull {}
impl RB for RangeInclusive<usize> {}
impl RB for RangeTo<usize> {}
impl RB for RangeToInclusive<usize> {}

impl<T: RB> From<T> for TensorIdx {
    fn from(range: T) -> Self {
        use std::ops::Bound::*;
        let start = match range.start_bound() {
            Included(idx) => Included(*idx),
            Excluded(idx) => Excluded(*idx),
            Unbounded => Unbounded,
        };
        let end = match range.end_bound() {
            Included(idx) => Included(*idx),
            Excluded(idx) => Excluded(*idx),
            Unbounded => Unbounded,
        };
        TensorIdx::Narrow(start, end)
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
