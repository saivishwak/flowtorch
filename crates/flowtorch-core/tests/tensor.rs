use flowtorch_core::{DType, Device, Tensor};

fn zeros(device: &Device) -> Result<(), ()> {
    let tensor = Tensor::zeros(&[5, 2], DType::F32, device)?;
    let shape = tensor.shape();
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], 2);
    println!("{:?}", tensor);
    assert_eq!(DType::F32, tensor.dtype());
    Ok(())
}

fn ones(device: &Device) -> Result<(), ()> {
    let tensor = Tensor::ones(&[5, 2], DType::F32, device)?;
    let shape = tensor.shape();
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], 2);
    println!("{:?}", tensor);
    assert_eq!(DType::F32, tensor.dtype());
    Ok(())
}

#[test]
fn test_zeros() {
    let device = Device::Cpu;
    let _ = zeros(&device);
}

#[test]
fn test_ones() {
    let device = Device::Cpu;
    let _ = ones(&device);
}
