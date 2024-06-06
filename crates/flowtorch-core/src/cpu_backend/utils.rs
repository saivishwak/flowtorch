// Helper function to compare vectors of any type
pub(super) fn compare_vecs<T: PartialEq>(
    vec1: &Vec<T>,
    vec2: &Vec<T>,
    vec1_offset: (usize, usize),
    vec2_offset: (usize, usize),
) -> bool {
    let vec1_start = vec1_offset.0;
    let vec1_end = vec1_start + vec1_offset.1;
    let vec2_start = vec2_offset.0;
    let vec2_end = vec2_start + vec2_offset.1;
    if vec1_end - vec1_start != vec2_end - vec2_start {
        return false;
    }
    &vec1[vec1_start..vec1_end] == &vec2[vec2_start..vec2_end]
}
