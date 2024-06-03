use flowtorch_core::{Device, Tensor};

#[test]
fn comparison() {
    let device = &Device::Cpu;
    assert_eq!(
        Tensor::new(&[0.0, 1.0, 2.0], &device).unwrap(),
        Tensor::new(&[0.0, 1.0, 2.0], &device).unwrap()
    );
    assert_ne!(
        Tensor::new(&[0.0, 2.0, 3.0], &device).unwrap(),
        Tensor::new(&[0.0, 1.0, 2.0], &device).unwrap()
    );
}

#[test]
fn add() {
    let device = &Device::Cpu;
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], &device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], &device).unwrap();
    let z = (x + y).unwrap();
    assert_eq!(Tensor::new(&[3.0f64, 4.0, 6.0], &device).unwrap(), z);

    //Int type
    let x = Tensor::new(&[1, 2, 3], &device).unwrap();
    let y = Tensor::new(&[2, 2, 3], &device).unwrap();
    let z = (x + y).unwrap();
    assert_eq!(Tensor::new(&[3, 4, 6], &device).unwrap(), z);
}

#[test]
fn mul() {
    let device = &Device::Cpu;
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], &device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], &device).unwrap();
    let z = (x * y).unwrap();
    assert_eq!(Tensor::new(&[2.0f64, 4.0, 9.0], &device).unwrap(), z);
}
