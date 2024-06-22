# FlowTorch
[![Latest version](https://img.shields.io/crates/v/flowtorch-core.svg)](https://crates.io/crates/flowtorch-core)
[![Documentation](https://docs.rs/flowtorch-core/badge.svg)](https://docs.rs/flowtorch-core)
![License](https://img.shields.io/crates/l/flowtorch-core.svg)
[![codecov](https://codecov.io/gh/saivishwak/flowtorch/graph/badge.svg?token=Q24EMJYUCD)](https://codecov.io/gh/saivishwak/flowtorch)

Safe and Performant Scientific Computing Library.

Right now the library is still in research phase where I am taking time to understand the internal implementations of Candle and Pytorch by reimplementing them. For production usage please try out [Candle](https://github.com/huggingface/candle) because its really awesome.

====

### Todo
- [x] Add Tensor::new method to create a Tensdor from data
- [x] Fix shape similar to PyTorch
- [x] Provide default contiguous layout - C type (Row Major)
- [x] Implement view and reshape in Layout
- [x] Add Proper Error Handling (https://github.com/dtolnay/thiserror)
- [x] Add Arc and Rw::Lock for Storage in Tensor
- [x] Implement Display for Tensor to print the matrix
- [x] Implement Indexing which return a tensor
- [x] Refactor indexing
- [x] Add tuple with range indexing
- [x] Fix Pretty Printing of Tensor
- [x] Fix Display with PrintOptions
- [x] Add I32 data type
- [x] Add Baisc Docs for Tensor
- [x] Add Tests for Narrow and Squeeze,
- [x] Add macros and optimze code for ops
- [x] Add Basic Binray Ops to Tensor
- [x] Add basic unary ops
- [x] Add Basic CUDA Device support
- [x] Add Basic CUDA Ops
- [x] Add to_device method support for converting the device of Tensor
- [x] Refactor Error::Unknown and fix todo
- [x] The Array trait implementation right now is doing many recurssive operations, can we minimize it?
- [x] Combine common tests for different devices
- [x] Add to_dtype method to convert the data type
- [ ] Support Fotran Memory layout (colum Major)
- [ ] Add Basic Back Prop
- [ ] Fix Comp Op
- [ ] Add in place Binary and Unary ops (ex, tensor.add_, tensor.abs_)
- [ ] Improve Code Coverage
- [ ] Add Tensor.to_vec methods, to get different dim vectors
- [ ] Fix select_index method
- [ ] Study Torch7 Library and how Storage, Tensor is implemented
- [ ] Add Benchmarking (https://github.com/bheisler/criterion.rs)

===

###  Future Todo
- [ ] flowtorch-nn crate for Neural Net Utilities and Layers
- [ ] Inference MNIST
- [ ] Inference ImageNet Model
- [ ] Inference BERT/LLM's
- [ ] Add basic autograd functionality
- [ ] Train and Test a simple MNIST example



##### Setup

Running Tests for CUDA
```sh
cargo test --features "cuda" -- --nocapture 
```

Running Coverage asdfasdf
```sh
# Without CUDA
cargo tarpaulin --verbose --workspace --timeout 120 --out Html
```

```sh
# With CUDA
cargo tarpaulin --features="cuda" --verbose --workspace --timeout 120 --out Html
```

To excule cuda_backend from coverage for non cuda environment (GithubActions)
```sh
cargo tarpaulin --verbose --workspace --timeout 120 --out Html --exclude-files "crates/flowtorch-core/src/cuda_backend/**/*"
```
