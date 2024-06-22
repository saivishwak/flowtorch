use flowtorch_core::{Device, IndexOp, StridedIndex, Tensor};

pub fn test_narrow(device: &Device) {
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

    let t1 = Tensor::new(&[[1, 2], [3, 4], [5, 6]], &device)
        .unwrap()
        .narrow(1, 0, 1)
        .unwrap();
    let t2 = Tensor::new(&[[1], [3], [5]], &device).unwrap();
    assert!(t1.equal(&t2));
}

pub fn squeeze(device: &Device) {
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

pub fn test_indexer(device: &Device) {
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
