# FlowTorch

FlowTorch: Safe and Performant Scientific Computing Library

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
- [ ] Add macros and optimze code for ops
- [ ] Add Basic Ops to Tensor along with Dotproduct
- [ ] Add to_dtype method to convert the data type
- [ ] Fix select_index method
- [ ] Add CUDA Device support
- [ ] Add to_device method support for converting the device of Tensor
- [ ] Add Tensor.to_vec methods, to get different dim vectors
- [ ] Support Fotran Memory layout (colum Major)
- [ ] Study Torch7 Library and how Storage, Tensor is implemented

===

###  Future Todo
- [ ] flowtorch-nn crate for Neural Net Utilities and Layers
- [ ] Inference MNIST
- [ ] Inference ImageNet Model
- [ ] Inference BERT/LLM's
- [ ] Add basic autograd functionality
- [ ] Train and Test a simple MNIST example