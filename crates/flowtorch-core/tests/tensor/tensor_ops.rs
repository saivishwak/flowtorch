use flowtorch_core::{Device, Tensor};

pub fn test_comparison(device: &Device) {
    assert!(Tensor::new(&[0.0, 1.0, 2.0], device)
        .unwrap()
        .equal(&Tensor::new(&[0.0, 1.0, 2.0], device).unwrap()));
    assert!(!Tensor::new(&[0.0, 2.0, 3.0], device)
        .unwrap()
        .equal(&Tensor::new(&[0.0, 1.0, 2.0], device).unwrap()));
}

pub fn test_add(device: &Device) {
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], device).unwrap();
    let z = Tensor::add_impl(&x, &y).unwrap();
    assert!(Tensor::new(&[3.0f64, 4.0, 6.0], device).unwrap().equal(&z));

    let z = (x + y).unwrap();
    assert!(Tensor::new(&[3.0f64, 4.0, 6.0], device).unwrap().equal(&z));

    //Int type
    let x = Tensor::new(&[1, 2, 3], device).unwrap();
    let y = Tensor::new(&[2, 2, 3], device).unwrap();
    let z = Tensor::add_impl(&x, &y).unwrap();
    assert!(Tensor::new(&[3, 4, 6], device).unwrap().equal(&z));

    let z = (x + y).unwrap();
    assert!(Tensor::new(&[3, 4, 6], device).unwrap().equal(&z));
}

pub fn test_sub(device: &Device) {
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], device).unwrap();
    let z = Tensor::sub_impl(&y, &x).unwrap();
    assert!(Tensor::new(&[1.0f64, 0.0, 0.0], device).unwrap().equal(&z));

    let z = (y - x).unwrap();
    assert!(Tensor::new(&[1.0f64, 0.0, 0.0], device).unwrap().equal(&z));
}

pub fn test_mul(device: &Device) {
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], device).unwrap();
    let z = Tensor::mul_impl(&x, &y).unwrap();
    assert!(Tensor::new(&[2.0f64, 4.0, 9.0], device).unwrap().equal(&z));

    let z = (x * y).unwrap();
    assert!(Tensor::new(&[2.0f64, 4.0, 9.0], device).unwrap().equal(&z));
}

pub fn test_max_min(device: &Device) {
    let x = Tensor::new(&[1.0f64, 3.0, 3.0], device).unwrap();
    let y = Tensor::new(&[2.0f64, 2.0, 3.0], device).unwrap();
    let z = Tensor::max(&x, &y).unwrap();
    assert!(Tensor::new(&[2.0f64, 3.0, 3.0], device).unwrap().equal(&z));

    let z = Tensor::min(&x, &y).unwrap();
    assert!(Tensor::new(&[1.0f64, 2.0, 3.0], device).unwrap().equal(&z));
}

pub fn test_unary(device: &Device) {
    let x = Tensor::new(&[1.0f64, 2.0, 3.0], device).unwrap();
    let y = Tensor::neg(&x).unwrap();
    assert!(Tensor::new(&[-1.0f64, -2.0, -3.0], device)
        .unwrap()
        .equal(&y));

    let y = Tensor::sqr(&x).unwrap();
    assert!(Tensor::new(&[1.0f64, 4.0, 9.0], device).unwrap().equal(&y));

    let x = Tensor::new(&[4.0f64, 9.0, 16.0], device).unwrap();
    let y = Tensor::sqrt(&x).unwrap();
    assert!(Tensor::new(&[2.0f64, 3.0, 4.0], device).unwrap().equal(&y));

    let x = Tensor::new(&[4.2f64, 9.1, 16.0], device).unwrap();
    let y = Tensor::ceil(&x).unwrap();
    assert!(Tensor::new(&[5.0f64, 10.0, 16.0], device)
        .unwrap()
        .equal(&y));

    let x = Tensor::new(&[4.2f64, 9.0, 15.8], device).unwrap();
    let y = Tensor::floor(&x).unwrap();
    assert!(Tensor::new(&[4.0f64, 9.0, 15.0], device).unwrap().equal(&y));
}
