use std::vec;

use flowtorch_core::{DType, Device, Shape, Tensor};

// TODO - Need to add to_vec method to be able to test the output
fn zeros(device: &Device, dtype: DType) -> Result<Tensor, ()> {
    let s: &[usize] = &[5, 2];
    let shape: Shape = s.into();
    let tensor = Tensor::zeros(shape, dtype, device)?;
    let shape = tensor.shape();
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], 2);
    Ok(tensor)
}

// TODO - Need to add to_vec method to be able to test the output
fn ones(device: &Device, dtype: DType) -> Result<Tensor, ()> {
    let tensor = Tensor::ones((3, 3, 2), dtype, device)?;
    let shape = tensor.shape();
    assert_eq!(shape[0], 3);
    assert_eq!(shape[1], 3);
    assert_eq!(shape[2], 2);
    Ok(tensor)
}

#[test]
fn test_zeros() {
    let device = Device::Cpu;
    let _ = zeros(&device, DType::F32);
}

#[test]
fn test_ones() {
    let device = Device::Cpu;
    let _ = ones(&device, DType::F32);
}

#[test]
fn strides() -> Result<(), ()> {
    let device = Device::Cpu;
    assert_eq!(
        Tensor::ones((3, 3, 2), DType::F32, &device)?.stride(),
        &[6, 2, 1]
    );

    assert_eq!(Tensor::ones((1, 2), DType::F32, &device)?.stride(), &[2, 1]);

    assert_eq!(Tensor::ones((4,), DType::F32, &device)?.stride(), &[1]);

    assert_eq!(Tensor::ones((), DType::F32, &device)?.stride(), &[]);

    Ok(())
}

#[test]
fn shape() -> Result<(), ()> {
    let device = Device::Cpu;
    assert_eq!(Tensor::new(&[1.0], &device)?.shape(), &[1]);

    assert_eq!(Tensor::new(&[1.0, 2.0, 3.0], &device)?.shape(), &[3]);

    assert_eq!(Tensor::new(&[[1.0], [3.0]], &device)?.shape(), &[2, 1]);

    assert_eq!(
        Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], &device)?.shape(),
        &[2, 2]
    );

    assert_eq!(
        Tensor::new(&[[[1.0], [2.0]], [[3.0], [4.0]]], &device)?.shape(),
        &[2, 2, 1]
    );

    assert_eq!(
        Tensor::new(
            &[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
            &device
        )?
        .shape(),
        &[2, 2, 2]
    );

    assert_eq!(
        Tensor::new(
            &[
                [[[1.0], [1.0]], [[2.0], [2.0]]],
                [[[3.0], [3.0]], [[4.0], [4.0]]]
            ],
            &device
        )?
        .shape(),
        &[2, 2, 2, 1]
    );

    assert_eq!(
        Tensor::new(
            &[
                [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],
                [[[3.0, 3.0], [3.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]]
            ],
            &device
        )?
        .shape(),
        &[2, 2, 2, 2]
    );

    assert_eq!(Tensor::new(vec![1.0], &device)?.shape(), &[1]);

    assert_eq!(Tensor::new(vec![1.0, 2.0, 3.0], &device)?.shape(), &[3]);

    assert_eq!(
        Tensor::new(vec![vec![1.0], vec![3.0]], &device)?.shape(),
        &[2, 1]
    );

    assert_eq!(
        Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)?.shape(),
        &[2, 2]
    );

    assert_eq!(
        Tensor::new(
            vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]],
            &device
        )?
        .shape(),
        &[2, 2, 1]
    );

    assert_eq!(
        Tensor::new(
            vec![
                vec![vec![1.0, 1.0], vec![2.0, 2.0]],
                vec![vec![3.0, 3.0], vec![4.0, 4.0]]
            ],
            &device
        )?
        .shape(),
        &[2, 2, 2]
    );

    assert_eq!(
        Tensor::new(
            vec![
                vec![vec![vec![1.0], vec![1.0]], vec![vec![2.0], vec![2.0]]],
                vec![vec![vec![3.0], vec![3.0]], vec![vec![4.0], vec![4.0]]]
            ],
            &device
        )?
        .shape(),
        &[2, 2, 2, 1]
    );

    assert_eq!(
        Tensor::new(
            vec![
                vec![
                    vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                    vec![vec![2.0, 2.0], vec![2.0, 2.0]]
                ],
                vec![
                    vec![vec![3.0, 3.0], vec![3.0, 3.0]],
                    vec![vec![4.0, 4.0], vec![4.0, 4.0]]
                ]
            ],
            &device
        )?
        .shape(),
        &[2, 2, 2, 2]
    );

    Ok(())
}

#[test]
fn dtype() -> Result<(), ()> {
    let device = Device::Cpu;

    assert_eq!(ones(&device, DType::F32)?.dtype(), DType::F32);
    assert_eq!(ones(&device, DType::F64)?.dtype(), DType::F64);
    assert_eq!(ones(&device, DType::I64)?.dtype(), DType::I64);
    assert_eq!(ones(&device, DType::U8)?.dtype(), DType::U8);
    assert_eq!(ones(&device, DType::U32)?.dtype(), DType::U32);

    assert_eq!(zeros(&device, DType::F32)?.dtype(), DType::F32);
    assert_eq!(zeros(&device, DType::F64)?.dtype(), DType::F64);
    assert_eq!(zeros(&device, DType::I64)?.dtype(), DType::I64);
    assert_eq!(zeros(&device, DType::U8)?.dtype(), DType::U8);
    assert_eq!(zeros(&device, DType::U32)?.dtype(), DType::U32);

    assert_eq!(Tensor::new(&[1.0], &device)?.dtype(), DType::F64);
    assert_eq!(Tensor::new(&[1.0f32], &device)?.dtype(), DType::F32);
    assert_eq!(Tensor::new(&[1i64], &device)?.dtype(), DType::I64);
    assert_eq!(Tensor::new(&[1u8], &device)?.dtype(), DType::U8);
    assert_eq!(Tensor::new(&[1u32], &device)?.dtype(), DType::U32);
    Ok(())
}

#[test]
// TODO - Need to add to_vec method to be able to test the output
fn new() {
    let _device = Device::Cpu;
    // println!(
    //     "Tensor from data {:?}",
    //     Tensor::new(
    //         &[[[0.0, 1.0], [3.0, 1.0]], [[1.0, 1.0], [3.0, 1.0]]],
    //         device
    //     )?
    // );
    // println!("Tensor from data {:?}", Tensor::new(&[1.0], device)?);
    // println!(
    //     "Tensor from data {:?}",
    //     Tensor::new(&[[[1.0], [2.0]], [[3.0], [2.0]]], device)?
    // );
    // println!(
    //     "Tensor from data {:?}",
    //     Tensor::new(vec![1.0, 2.0], device)?
    // );
    // println!(
    //     "Tensor from data {:?}",
    //     Tensor::new(vec![vec![2.0, 10.0], vec![2.0, 4.0]], device)?
    // );

    // println!(
    //     "Tensor from data {:?}",
    //     Tensor::new(vec![vec![2.0, 10.0], vec![2.0, 4.0]], device)?.reshape((1, 4))?
    // );
    // assert_eq!(
    //     Tensor::new(vec![1, 2], DType::F32, device)?.to_vec(),
    //     vec![1, 2]
    // );
}
