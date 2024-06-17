use flowtorch_core::{
    error::{Error, LayoutError},
    DType, Device, Tensor,
};

#[test]
fn init_error() {
    let device = &Device::Cpu;
    let e = Tensor::from_vec(vec![0.0, 2.0, 3.0], (2, 2), device)
        .err()
        .unwrap();
    assert_eq!(
        e,
        Error::TensorInit(
            Some(DType::F64),
            "Provided shape and length of Data does not match".to_string()
        )
    );
}

#[test]
fn layout_error() {
    let device = &Device::Cpu;
    let e = Tensor::from_vec(vec![0.0, 2.0, 3.0], 3, device)
        .unwrap()
        .reshape((3, 3))
        .err()
        .unwrap();
    assert_eq!(
        e,
        LayoutError::ReshapeError("Mismatch in Elements".to_string()).into()
    );
}
