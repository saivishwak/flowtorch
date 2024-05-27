use std::vec;

use flowtorch_core::{DType, Device, Shape, Tensor};

// TODO - Need to add to_vec method to be able to test the output
fn zeros(device: &Device, dtype: DType) -> Result<Tensor, ()> {
    let s: &[usize] = &[5, 2];
    let shape: Shape = s.into();
    let tensor = Tensor::zeros(shape, dtype, device).unwrap();
    let shape = tensor.dims();
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], 2);
    Ok(tensor)
}

// TODO - Need to add to_vec method to be able to test the output
fn ones(device: &Device, dtype: DType) -> Result<Tensor, ()> {
    let tensor = Tensor::ones((3, 3, 2), dtype, device).unwrap();
    let shape = tensor.dims();
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
        Tensor::ones((3, 3, 2), DType::F32, &device)
            .unwrap()
            .stride(),
        &[6, 2, 1]
    );

    assert_eq!(
        Tensor::ones((1, 2), DType::F32, &device).unwrap().stride(),
        &[2, 1]
    );

    assert_eq!(
        Tensor::ones((4,), DType::F32, &device).unwrap().stride(),
        &[1]
    );
    let shape: &[usize] = &[4];
    assert_eq!(
        Tensor::ones(shape, DType::F32, &device).unwrap().stride(),
        &[1]
    );
    assert_eq!(
        Tensor::ones(vec![4], DType::F32, &device).unwrap().stride(),
        &[1]
    );

    assert_eq!(Tensor::ones((), DType::F32, &device).unwrap().stride(), &[]);

    assert_eq!(Tensor::new(1.0, &device).unwrap().stride(), &[]);
    assert_eq!(Tensor::new(&[1.0], &device).unwrap().stride(), &[1]);
    assert_eq!(
        Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], &device)
            .unwrap()
            .stride(),
        &[2, 1]
    );

    Ok(())
}

#[test]
fn shape() -> Result<(), ()> {
    let device = Device::Cpu;
    assert_eq!(Tensor::new(1.0, &device).unwrap().dims(), &[]);
    assert_eq!(Tensor::new(&[1.0], &device).unwrap().dims(), &[1]);

    assert_eq!(Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap().dims(), &[3]);

    assert_eq!(
        Tensor::new(&[[1.0], [3.0]], &device).unwrap().dims(),
        &[2, 1]
    );

    assert_eq!(
        Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], &device)
            .unwrap()
            .dims(),
        &[2, 2]
    );

    assert_eq!(
        Tensor::new(&[[[1.0], [2.0]], [[3.0], [4.0]]], &device)
            .unwrap()
            .dims(),
        &[2, 2, 1]
    );

    assert_eq!(
        Tensor::new(
            &[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
            &device
        )
        .unwrap()
        .dims(),
        &[2, 2, 2]
    );

    assert_eq!(
        Tensor::new(
            &[
                [[[1.0], [1.0]], [[2.0], [2.0]]],
                [[[3.0], [3.0]], [[4.0], [4.0]]]
            ],
            &device
        )
        .unwrap()
        .dims(),
        &[2, 2, 2, 1]
    );

    assert_eq!(
        Tensor::new(
            &[
                [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],
                [[[3.0, 3.0], [3.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]]
            ],
            &device
        )
        .unwrap()
        .dims(),
        &[2, 2, 2, 2]
    );

    assert_eq!(Tensor::new(vec![1.0], &device).unwrap().dims(), &[1]);

    assert_eq!(
        Tensor::new(vec![1.0, 2.0, 3.0], &device).unwrap().dims(),
        &[3]
    );

    assert_eq!(
        Tensor::new(vec![vec![1.0], vec![3.0]], &device)
            .unwrap()
            .dims(),
        &[2, 1]
    );

    assert_eq!(
        Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device)
            .unwrap()
            .dims(),
        &[2, 2]
    );

    assert_eq!(
        Tensor::new(
            vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]],
            &device
        )
        .unwrap()
        .dims(),
        &[2, 2, 1]
    );

    assert_eq!(
        Tensor::new(
            vec![
                vec![vec![1.0, 1.0], vec![2.0, 2.0]],
                vec![vec![3.0, 3.0], vec![4.0, 4.0]]
            ],
            &device
        )
        .unwrap()
        .dims(),
        &[2, 2, 2]
    );

    assert_eq!(
        Tensor::new(
            vec![
                vec![vec![vec![1.0], vec![1.0]], vec![vec![2.0], vec![2.0]]],
                vec![vec![vec![3.0], vec![3.0]], vec![vec![4.0], vec![4.0]]]
            ],
            &device
        )
        .unwrap()
        .dims(),
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
        )
        .unwrap()
        .dims(),
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

    assert_eq!(Tensor::new(1.0, &device).unwrap().dtype(), DType::F64);
    assert_eq!(Tensor::new(&[1.0], &device).unwrap().dtype(), DType::F64);
    assert_eq!(Tensor::new(&[1.0f32], &device).unwrap().dtype(), DType::F32);
    assert_eq!(Tensor::new(&[1i64], &device).unwrap().dtype(), DType::I64);
    assert_eq!(Tensor::new(&[1u8], &device).unwrap().dtype(), DType::U8);
    assert_eq!(Tensor::new(&[1u32], &device).unwrap().dtype(), DType::U32);
    Ok(())
}

#[test]
// TODO - Need to add to_vec method to be able to test the output
fn format() {
    let device = &Device::Cpu;
    assert_eq!(
        "[0]",
        Tensor::from_vec(vec![0i64], (1,), device)
            .unwrap()
            .as_string(Some(false))
            .unwrap()
            .as_str()
    );
    assert_eq!(
        "[1]",
        Tensor::from_vec(vec![1.0], (1,), device)
            .unwrap()
            .as_string(Some(false))
            .unwrap()
            .as_str()
    );
    assert_eq!(
        "[1, 2, 3]",
        Tensor::new(&[1.0, 2.0, 3.0], device)
            .unwrap()
            .as_string(Some(false))
            .unwrap()
            .as_str()
    );
    assert_eq!(
        "[[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]], [[3, 2, 2], [3, 2, 2], [3, 2, 2]]]",
        Tensor::new(
            &[
                [[1.0f32, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                [[3.0, 2.0, 2.0], [3.0, 2.0, 2.0], [3.0, 2.0, 2.0]]
            ],
            device
        )
        .unwrap()
        .as_string(Some(false))
        .unwrap()
        .as_str()
    );
    assert_eq!(
        "[[[1], [1]]]",
        Tensor::new(&[[[1.0f32], [1.0]]], device)
            .unwrap()
            .as_string(Some(false))
            .unwrap()
            .as_str()
    );
    assert_eq!(
        "[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]], [[[2], [2], [2]], [[2], [2], [2]], [[2], [2], [2]]], [[[3], [2], [2]], [[3], [2], [2]], [[3], [2], [2]]]]",
        Tensor::new(
            &[
                [
                    [[1.0f32], [1.0], [1.0]],
                    [[1.0], [1.0], [1.0]],
                    [[1.0], [1.0], [1.0]]
                ],
                [
                    [[2.0], [2.0], [2.0]],
                    [[2.0], [2.0], [2.0]],
                    [[2.0], [2.0], [2.0]]
                ],
                [
                    [[3.0], [2.0], [2.0]],
                    [[3.0], [2.0], [2.0]],
                    [[3.0], [2.0], [2.0]]
                ]
            ],
            device
        )
        .unwrap()
        .as_string(Some(false))
        .unwrap()
        .as_str()
    );
}
