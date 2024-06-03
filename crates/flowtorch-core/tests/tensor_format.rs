use flowtorch_core::{DType, Device, IndexOp, PrintOptions, PrintProfiles, Tensor};

#[test]
fn format_no_treshold() {
    let mut print_options = PrintOptions::new();
    print_options.set_precision(0);
    let device = &Device::Cpu;
    assert_eq!(
        "[0]",
        Tensor::from_vec(vec![0i64], (1,), device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
            .as_str()
    );
    assert_eq!(
        "[1]",
        Tensor::from_vec(vec![1.0], (1,), device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
            .as_str()
    );
    assert_eq!(
        "[1, 2, 3]",
        Tensor::new(&[1.0, 2.0, 3.0], device)
            .unwrap()
            .fmt(Some(print_options.clone()))
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
        .fmt(Some(print_options.clone()))
        .unwrap()
        .as_str()
    );
    assert_eq!(
        "[[[1], [1]]]",
        Tensor::new(&[[[1.0f32], [1.0]]], device)
            .unwrap()
            .fmt(Some(print_options.clone()))
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
        .fmt(Some(print_options.clone()))
        .unwrap()
        .as_str()
    );

    assert_eq!(
        "[0, 0, 0]",
        Tensor::zeros(3, DType::F64, device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
    );

    assert_eq!(
        "[1, 1, 1]",
        Tensor::ones(3, DType::F64, device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
    );

    assert_eq!(
        "[[1], [1], [1]]",
        Tensor::ones((3, 1), DType::F64, device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
    );

    assert_eq!(
        "[[0], [0], [0]]",
        Tensor::zeros((3, 1), DType::F64, device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
    );

    assert_eq!(
        "[[[0, 0]], [[0, 0]], [[0, 0]]]",
        Tensor::zeros((3, 1, 2), DType::F64, device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
    );

    assert_eq!(
        "[[[1, 1]], [[1, 1]], [[1, 1]]]",
        Tensor::ones((3, 1, 2), DType::F64, device)
            .unwrap()
            .fmt(Some(print_options.clone()))
            .unwrap()
    );
}

#[test]
fn format_treshold() {
    let mut print_options = PrintOptions::new();
    print_options.set_precision(0);
    print_options.set_threshold(5);
    let device = &Device::Cpu;

    assert_eq!(
        "[[[1, 1]], [[1, 1]], [[1, 1]],  ... , [[1, 1]], [[1, 1]], [[1, 1]]]",
        Tensor::new(
            &[
                [[1i64, 1]],
                [[1, 1]],
                [[1, 1]],
                [[1, 1]],
                [[1, 1]],
                [[1, 1]],
                [[1, 1]],
                [[1, 1]],
                [[1, 1]]
            ],
            device
        )
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    assert_eq!(
        "[[[1, 1, 1,  ... , 1, 1, 1]], [[1, 1, 1,  ... , 1, 1, 1]], [[1, 1, 1,  ... , 1, 1, 1]],  ... , [[1, 1, 1,  ... , 1, 1, 1]], [[1, 1, 1,  ... , 1, 1, 1]], [[1, 1, 1,  ... , 1, 1, 1]]]",
        Tensor::new(
            &[
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]],
                [[1i64, 1, 1, 1, 1, 1, 1, 1, 1]]
            ],
            device
        )
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    assert_eq!(
        "[1, 1, 1,  ... , 1, 1, 1]",
        Tensor::new(
            &[
                [1i64, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            ],
            device
        )
        .unwrap()
        .i(0)
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    assert_eq!(
        "[2, 2, 2,  ... , 2, 2, 2]",
        Tensor::new(
            &[
                [1i64, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            ],
            device
        )
        .unwrap()
        .i(1)
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    assert_eq!(
        "2",
        Tensor::new(
            &[
                [1i64, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            ],
            device
        )
        .unwrap()
        .i((1, 0))
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );
}

#[test]
fn format_precision() {
    let mut print_options = PrintOptions::new();
    print_options.set_precision(1);
    print_options.set_threshold(5);
    let device = &Device::Cpu;

    assert_eq!(
        "[[[1.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]],  ... , [[1.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]]]",
        Tensor::new(
            &[
                [[1f64, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]]
            ],
            device
        )
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    assert_eq!(
        "[2.0, 2.0]",
        Tensor::new(
            &[
                [[1f64, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]]
            ],
            device
        )
        .unwrap()
        .i((3, 0))
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    print_options.set_precision(2);
    assert_eq!(
        "[1.00, 1.00, 1.00,  ... , 1.00, 1.00, 1.00]",
        Tensor::new(
            &[
                [1f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            ],
            device
        )
        .unwrap()
        .i(0)
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    print_options.set_precision(3);
    assert_eq!(
        "[2.000, 2.000, 2.000,  ... , 2.000, 2.000, 2.000]",
        Tensor::new(
            &[
                [1f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            ],
            device
        )
        .unwrap()
        .i(1)
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    assert_eq!(
        "2.000",
        Tensor::new(
            &[
                [1f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            ],
            device
        )
        .unwrap()
        .i((1, 0))
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );
}

#[test]
fn format_profiles() {
    //Default
    let mut print_options = PrintOptions::new();
    let device = &Device::Cpu;

    assert_eq!(
        "2.0000",
        Tensor::new(
            &[
                [1f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            ],
            device
        )
        .unwrap()
        .i((1, 0))
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    print_options.set_profile(PrintProfiles::Short);
    assert_eq!(
        "[[[1.00, 1.00]], [[1.00, 1.00]], [[1.00, 1.00]], [[1.00, 1.00]], [[1.00, 1.00]], [[1.00, 1.00]]]",
        Tensor::new(
            &[
                [[1f64, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
            ],
            device
        )
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );

    print_options.set_profile(PrintProfiles::Full);
    assert_eq!(
        "[[[1.0000, 1.0000]], [[1.0000, 1.0000]], [[1.0000, 1.0000]], [[1.0000, 1.0000]], [[1.0000, 1.0000]], [[1.0000, 1.0000]]]",
        Tensor::new(
            &[
                [[1f64, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
            ],
            device
        )
        .unwrap()
        .fmt(Some(print_options.clone()))
        .unwrap()
    );
}
