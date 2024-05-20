# FlowTorch

FlowTorch: Safe and Performant Scientific Computing Library

====

### Todo

- [x] Add Tensor::new method to create a Tensdor from data
- [x] Fix shape similar to PyTorch
- [x] Provide default contiguous layout - C type (Row Major)
- [ ] Add Proper Error Handling (https://github.com/dtolnay/thiserror)
- [ ] Add Tensor.to_vec methods, to get different dim vectors
- [ ] Implement view and reshape in Layout
- [ ] Add Arc and Rw::Lock for Storage in Tensor
- [ ] Add Basic Ops to Tensor along with Dotproduct
- [ ] Prove Fotran Memory layout (colum Major)


===

###  Future Todo
- [ ] flowtorch-nn crate for Neural Net Utilities and Layers
- [ ] Inference MNIST
- [ ] Inference ImageNet Model
- [ ] Inference BERT/LLM's
- [ ] Add basic autograd functionality
- [ ] Train and Test a simple MNIST example