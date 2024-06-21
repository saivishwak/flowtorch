use flowtorch_core::{
    error::{Error, LayoutError},
    DType, Device, Tensor,
};

pub fn test_error(device: &Device) {
    println!("Running Error Combined Tests");
    init_error(device);
    layout_error(device);
}

fn init_error(device: &Device) {
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

fn layout_error(device: &Device) {
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
