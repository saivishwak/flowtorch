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
    let z = Tensor::add(&x, &y).unwrap();
    assert_eq!(Tensor::new(&[3.0f64, 4.0, 6.0], &device).unwrap(), z);

    let z = (x + y).unwrap();
    assert_eq!(Tensor::new(&[3.0f64, 4.0, 6.0], &device).unwrap(), z);

    //Int type
    let x = Tensor::new(&[1, 2, 3], &device).unwrap();
    let y = Tensor::new(&[2, 2, 3], &device).unwrap();
    let z = Tensor::add(&x, &y).unwrap();
    assert_eq!(Tensor::new(&[3, 4, 6], &device).unwrap(), z);

    let z = (x + y).unwrap();
    assert_eq!(Tensor::new(&[3, 4, 6], &device).unwrap(), z);
}

#[test]
fn sub() {
    let device = &Device::Cpu;
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], &device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], &device).unwrap();
    let z = Tensor::sub(&y, &x).unwrap();
    assert_eq!(Tensor::new(&[1.0f64, 0.0, 0.0], &device).unwrap(), z);

    let z = (y - x).unwrap();
    assert_eq!(Tensor::new(&[1.0f64, 0.0, 0.0], &device).unwrap(), z);
}

#[test]
fn mul() {
    let device = &Device::Cpu;
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], &device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], &device).unwrap();
    let z = Tensor::mul(&x, &y).unwrap();
    assert_eq!(Tensor::new(&[2.0f64, 4.0, 9.0], &device).unwrap(), z);

    let z = (x * y).unwrap();
    assert_eq!(Tensor::new(&[2.0f64, 4.0, 9.0], &device).unwrap(), z);
}

#[test]
fn max_min() {
    let device = &Device::Cpu;
    let x = Tensor::new(&[1.0f64, 3.0, 3.0], &device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], &device).unwrap();
    let z = Tensor::max(&x, &y).unwrap();
    assert_eq!(Tensor::new(&[2.0f64, 3.0, 3.0], &device).unwrap(), z);

    let z = Tensor::min(&x, &y).unwrap();
    assert_eq!(Tensor::new(&[1.0f64, 2.0, 3.0], &device).unwrap(), z);
}

#[test]
fn unary() {
    let device = &Device::Cpu;
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], &device).unwrap();
    let y = Tensor::neg(&x).unwrap();
    assert_eq!(Tensor::new(&[-1.0f64, -2.0, -3.0], &device).unwrap(), y);

    let y = Tensor::sqr(&x).unwrap();
    assert_eq!(Tensor::new(&[1.0f64, 4.0, 9.0], &device).unwrap(), y);

    let x = Tensor::new(&[4.0f64, 9.0, 16.0], &device).unwrap();
    let y = Tensor::sqrt(&x).unwrap();
    assert_eq!(Tensor::new(&[2.0f64, 3.0, 4.0], &device).unwrap(), y);

    let x = Tensor::new(&[4.2f64, 9.1, 16.0], &device).unwrap();
    let y = Tensor::ceil(&x).unwrap();
    assert_eq!(Tensor::new(&[5.0f64, 10.0, 16.0], &device).unwrap(), y);

    let x = Tensor::new(&[4.2f64, 9.0, 15.8], &device).unwrap();
    let y = Tensor::floor(&x).unwrap();
    assert_eq!(Tensor::new(&[4.0f64, 9.0, 15.0], &device).unwrap(), y);
}
