# FlowTorch

FlowTorch: Safe and Performant Scientific Computing Library

====

### Todo

- [x] Add Tensor::new method to create a Tensdor from data
- [x] Fix shape similar to PyTorch
- [x] Provide default contiguous layout - C type (Row Major)
- [x] Implement view and reshape in Layout
- [x] Add Proper Error Handling (https://github.com/dtolnay/thiserror)
- [ ] Do we need to impl indexing and tensor data modification like in pytorch?
- [ ] Add to_dtype method to convert the data type
- [ ] Add Tensor.to_vec methods, to get different dim vectors
- [ ] Add Basic Ops to Tensor along with Dotproduct
- [ ] Add CUDA Device support
- [ ] Add to_device method support for converting the device of Tensor
- [ ] Add Arc and Rw::Lock for Storage in Tensor
- [ ] Prove Fotran Memory layout (colum Major)


===

###  Future Todo
- [ ] flowtorch-nn crate for Neural Net Utilities and Layers
- [ ] Inference MNIST
- [ ] Inference ImageNet Model
- [ ] Inference BERT/LLM's
- [ ] Add basic autograd functionality
- [ ] Train and Test a simple MNIST example