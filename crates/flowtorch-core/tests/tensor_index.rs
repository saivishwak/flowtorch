use flowtorch_core::{Device, IndexOp, Tensor};

#[test]
fn indexer() {
    let device = &Device::Cpu;
    assert_eq!(
        Tensor::new(&[0.0, 1.0, 2.0], &device)
            .unwrap()
            .i(1)
            .unwrap(),
        Tensor::new(1.0, &device).unwrap()
    );
    //Check Data type
    assert_ne!(
        Tensor::new(&[0.0f32, 1.0, 2.0], &device)
            .unwrap()
            .i(1)
            .unwrap(),
        Tensor::new(1.0f64, &device).unwrap()
    );
    assert_eq!(
        Tensor::new(vec![vec![vec![&[0.0], &[1.0], &[2.0]]]], &device)
            .unwrap()
            .i((0, 0, 1, 0))
            .unwrap(),
        Tensor::new(1.0, &device).unwrap()
    );
    assert_eq!(
        Tensor::new(vec![vec![vec![&[0.0], &[1.0], &[2.0]]]], &device)
            .unwrap()
            .i((0, 0, 2, 0))
            .unwrap(),
        Tensor::new(2.0, &device).unwrap()
    );
}
