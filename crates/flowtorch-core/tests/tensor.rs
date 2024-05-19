use std::vec;

use flowtorch_core::{DType, Device, Shape, Tensor};

fn zeros(device: &Device) -> Result<(), ()> {
    let s: &[usize] = &[5, 2];
    let shape: Shape = s.into();
    let tensor = Tensor::zeros(shape, DType::F32, device)?;
    let shape = tensor.shape();
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], 2);
    assert_eq!(DType::F32, tensor.dtype());
    Ok(())
}

fn ones(device: &Device) -> Result<(), ()> {
    let tensor = Tensor::ones((3, 3, 2), DType::F32, device)?;
    let shape = tensor.shape();
    assert_eq!(shape[0], 3);
    assert_eq!(shape[1], 3);
    assert_eq!(shape[2], 2);
    assert_eq!(DType::F32, tensor.dtype());
    Ok(())
}

fn strides(device: &Device) -> Result<(), ()> {
    assert_eq!(
        Tensor::ones((3, 3, 2), DType::F32, device)?.strides(),
        &[6, 2, 1]
    );

    assert_eq!(Tensor::ones((1, 2), DType::F32, device)?.strides(), &[2, 1]);

    assert_eq!(Tensor::ones((4,), DType::F32, device)?.strides(), &[1]);

    assert_eq!(Tensor::ones((), DType::F32, device)?.strides(), &[]);

    Ok(())
}

fn new(device: &Device) -> Result<(), ()> {
    println!(
        "Tensor from data {:?}",
        Tensor::new(
            &[[[0.0, 1.0], [3.0, 1.0]], [[1.0, 1.0], [3.0, 1.0]]],
            device
        )?
    );
    println!("Tensor from data {:?}", Tensor::new(&[1.0], device)?);
    println!(
        "Tensor from data {:?}",
        Tensor::new(&[[[1.0], [2.0]], [[3.0], [2.0]]], device)?
    );
    println!(
        "Tensor from data {:?}",
        Tensor::new(vec![1.0, 2.0], device)?
    );
    println!(
        "Tensor from data {:?}",
        Tensor::new(vec![vec![2.0, 10.0], vec![2.0, 4.0]], device)?
    );
    // assert_eq!(
    //     Tensor::new(vec![1, 2], DType::F32, device)?.to_vec(),
    //     vec![1, 2]
    // );
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

#[test]
fn test_strides() {
    let device = Device::Cpu;
    let _ = strides(&device);
}

#[test]
fn test_new() {
    let device = Device::Cpu;
    let _ = new(&device);
}
