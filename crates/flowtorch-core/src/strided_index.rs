use crate::layout::Layout;

#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_storage_index: Option<usize>,
    // Current index in each dim
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset)
        };
        StridedIndex {
            next_storage_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }

    pub fn from_layout(l: &'a Layout) -> Self {
        Self::new(l.dims_slice(), l.stride_slice(), l.start_offset())
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    //We calculate the next storage index and return the the current storage index
    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = match self.next_storage_index {
            None => return None,
            Some(storage_index) => storage_index,
        };
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, dim_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.dims.iter())
            .zip(self.stride.iter())
            .rev()
        {
            // Here we are checking if the current index exceeded the max number of elements in that dim
            // If yes then we go to the next outer dim and calculate the index based on stride
            let next_i = *multi_i + 1;
            if next_i < *dim_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += stride_i;
                break;
            } else {
                next_storage_index -= *multi_i * stride_i;
                *multi_i = 0
            }
        }
        self.next_storage_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}
