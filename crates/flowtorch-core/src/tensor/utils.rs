use super::constants;

pub fn can_truncate(truncate: Option<bool>, data_len: usize) -> (bool, usize) {
    let should_truncate = if let Some(t) = truncate {
        t
    } else {
        if data_len >= constants::TRUNCATION_LIMIT {
            true
        } else {
            false
        }
    };
    (should_truncate, constants::TRUNCATION_START_OFFSET)
}

/**
*   Returns a formatted string representation of the data like
*   [[[1, 1, 1], [1, 1, 1], .... [3, 2, 2], [3, 2, 2]]], the truncation detection is automatic if not provided.

    We can go crazy and do more formatting like new lines, for now this looks fine.

*/
pub fn as_string<T: std::fmt::Display>(
    data: &Vec<T>,
    dims: Vec<usize>,
    strides: Vec<usize>,
    truncate: Option<bool>,
    initial_offset: usize,
) -> String {
    let mut result = String::new();
    let dims: Vec<_> = dims.iter().rev().collect(); //Reverse the dimensions order so that we construct the innermost dimension first
    let strides: Vec<_> = strides.iter().rev().collect();

    if data.len() == 0 {
        return result;
    }

    //Scalar value
    if dims.len() == 0 {
        result.push_str(format!("{}", data[initial_offset]).as_str());
        return result;
    }
    let last_dim = dims[0];

    for _ in 0..dims.len() {
        result.push_str("[");
    }

    let mut sub_array_end_detected: bool;
    let mut truncated = false;
    let (should_truncate, truncation_start_offset) = can_truncate(truncate, data.len());

    let total_num_elems: usize = dims.iter().map(|d| *d).product::<usize>();
    let mut i = initial_offset;
    while i < data.len() && i < total_num_elems {
        result.push_str(format!("{}", data[i]).as_str());

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

            //Check if we exeeded truncation limit and should truncate
            if should_truncate && (i + 1) >= truncation_start_offset && !truncated {
                // This part is when we detected a subarray and truncation is needed we need to move to last second subarray
                if sub_array_end_detected {
                    result.push_str(".... [");
                    i = data.len() - last_dim * 2; // Skip to the last second subarray
                } else {
                    /* When the subarray is not detected, means is a 1D data and the check is needed as
                       we need to go to next subarray if not 1D, this checks basically if there can be subarrays in the data
                    */

                    //There is not subarray in the data its a 1D contiguous data
                    if data.len() / last_dim <= 1 {
                        result.push_str(".... ");
                        i = data.len() - 2; // Skip to the last second element
                    } else {
                        //There are subarrays in the data
                        i += 1;
                        continue;
                    }
                }
                truncated = true;
                continue;
            } else {
                if sub_array_end_detected {
                    for _k in 0..dim_end_len {
                        result.push_str("[");
                    }
                }
            }
        }
        i += 1;
    }

    return result;
}
