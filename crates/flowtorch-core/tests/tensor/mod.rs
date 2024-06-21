use flowtorch_core::{DType, Device};
mod error;
mod tensor_basic;
mod tensor_dtype;
mod tensor_format;
mod tensor_index;
mod tensor_ops;

pub fn test_tensor(device: &Device) {
    println!("Running Tensor Combined Tests");
    //Tensor Basic
    tensor_basic::test_ones(device);
    tensor_basic::test_zeros(device);
    let _ = tensor_basic::test_strides(device);
    let _ = tensor_basic::test_shape(device);
    let _ = tensor_basic::test_dtype(device);

    // TODO FOR CUDA
    // Tensor Dtype
    // tensor_dtype::test_comparison(device);

    //Tensor Format
    tensor_format::test_format_no_treshold(device);
    tensor_format::test_format_precision(device);
    tensor_format::test_format_profiles(device);
    tensor_format::test_format_treshold(device);

    //Tensor Indexer
    tensor_index::squeeze(device);
    tensor_index::test_indexer(device);
    tensor_index::test_narrow(device);

    //Tensor Ops
    tensor_ops::test_add(device);
    tensor_ops::test_comparison(device);
    // TODO FOR CUDA
    // tensor_ops::test_max_min(device);
    tensor_ops::test_mul(device);
    tensor_ops::test_sub(device);
    tensor_ops::test_unary(device);

    //Test Errors
    error::test_error(device);
}
