use crate::{layout::Layout, StridedIndex};

//TODO Need to implement Strided Indexing instead of plain vector index
// Helper function to compare vectors of any type
pub(super) fn compare_vecs<T: PartialEq>(
    vec1: &[T],
    vec2: &[T],
    vec1_layout: &Layout,
    vec2_layout: &Layout,
) -> bool {
    let strided_index_v1 = StridedIndex::from_layout(vec1_layout);
    let strided_index_v2 = StridedIndex::from_layout(vec2_layout);

    // Use zip and all to iterate through elements and check condition
    let result = strided_index_v1.zip(strided_index_v2).all(|(i1, i2)| {
        // Ensure indices are within bounds
        if i1 < vec1.len() && i2 < vec2.len() {
            // Compare elements from vec1 and vec2
            vec1[i1] == vec2[i2]
        } else {
            false
        }
    });
    result
}
