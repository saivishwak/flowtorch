use flowtorch_core::{DType, Device, Tensor};

pub fn test_comparison(device: &Device) {
    let t1 = Tensor::new(&[0.0f64, 1.0, 2.0], device).unwrap();
    let t2 = t1.to_dtype(DType::F32).unwrap();
    println!("{}", t2);
}
