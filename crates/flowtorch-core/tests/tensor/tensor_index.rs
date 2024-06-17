use flowtorch_core::{Device, IndexOp, Tensor};

#[test]
fn narrow() {
    let device = &Device::Cpu;
    assert!(Tensor::new(&[0.0, 1.0, 2.0], device)
        .unwrap()
        .narrow(0, 0, 2)
        .unwrap()
        .equal(&Tensor::new(&[0.0, 1.0], device).unwrap()));

    assert!(Tensor::new(&[[0.0], [1.0], [2.0]], device)
        .unwrap()
        .narrow(0, 0, 2)
        .unwrap()
        .equal(&Tensor::new(&[[0.0], [1.0]], device).unwrap()));

    //TODO - Check why this is failing
    // assert_eq!(
    //     Tensor::new(&[[1, 1], [2, 2], [3, 3]], &device)
    //         .unwrap()
    //         .narrow(1, 0, 1)
    //         .unwrap(),
    //     Tensor::new(&[[1], [2], [3]], &device).unwrap()
    // );
}

#[test]
fn squeeze() {
    let device = &Device::Cpu;
    assert!(Tensor::new(&[0.0, 1.0, 2.0], device)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .equal(&Tensor::new(&[0.0, 1.0, 2.0], device).unwrap()));

    assert!(Tensor::new(&[[0.0], [1.0], [2.0]], device)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .equal(&Tensor::new(&[0.0, 1.0, 2.0], device).unwrap()));
}

#[test]
fn indexer() {
    let device = &Device::Cpu;
    assert!(Tensor::new(&[0.0, 1.0, 2.0], device)
        .unwrap()
        .i(1)
        .unwrap()
        .equal(&Tensor::new(1.0, device).unwrap()),);
    //Check Data type
    assert!(!Tensor::new(&[0.0f32, 1.0, 2.0], device)
        .unwrap()
        .i(1)
        .unwrap()
        .equal(&Tensor::new(1.0f64, device).unwrap()));
    assert!(
        Tensor::new(vec![vec![vec![&[0.0], &[1.0], &[2.0]]]], device)
            .unwrap()
            .i((0, 0, 1, 0))
            .unwrap()
            .equal(&Tensor::new(1.0, device).unwrap())
    );
    assert!(
        Tensor::new(vec![vec![vec![&[0.0], &[1.0], &[2.0]]]], device)
            .unwrap()
            .i((0, 0, 2, 0))
            .unwrap()
            .equal(&Tensor::new(2.0, device).unwrap())
    );
}
