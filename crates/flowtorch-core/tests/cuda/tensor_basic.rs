#![allow(unused_imports)]
use flowtorch_core::{DType, Device, DeviceT, Tensor};

#[cfg(feature = "cuda")]
#[test]
fn basic() {
    let device = Device::new(DeviceT::Cuda(0)).unwrap();
    let tensor = Tensor::ones((3, 3, 2), DType::F64, &device).unwrap();
    let shape = tensor.dims();
    assert_eq!(shape[0], 3);
    assert_eq!(format!("{}", tensor.device()), "Cuda(0)");

    let tensor = Tensor::zeros((2, 2), DType::F64, &device).unwrap();
    assert_eq!(tensor.dims(), vec![2, 2]);

    let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
    assert_eq!(tensor.dims(), vec![3]);
}
