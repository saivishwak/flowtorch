use flow_core::{DType, Device, Tensor};

#[test]
fn zeros() {
    let tensor = Tensor::zeros(&[5, 2], DType::F32, Device::Cpu);
    let shape = tensor.shape();
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], 2);
}
