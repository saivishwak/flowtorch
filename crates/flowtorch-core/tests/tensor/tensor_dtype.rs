use flowtorch_core::{DType, Device, Tensor};

pub fn test_to_dtype(device: &Device) {
    let t1 = Tensor::new(&[0.0f64, 1.0, 2.0], device).unwrap();
    assert_eq!(t1.to_dtype(DType::F32).unwrap().dtype(), DType::F32);
    assert_eq!(t1.to_dtype(DType::I64).unwrap().dtype(), DType::I64);
    assert_eq!(t1.to_dtype(DType::I32).unwrap().dtype(), DType::I32);
    assert_eq!(t1.to_dtype(DType::U32).unwrap().dtype(), DType::U32);
    assert_eq!(t1.to_dtype(DType::U8).unwrap().dtype(), DType::U8);

    let t1 = Tensor::new(&[0.0f32, 1.0, 2.0], device).unwrap();
    assert_eq!(t1.to_dtype(DType::F64).unwrap().dtype(), DType::F64);
    assert_eq!(t1.to_dtype(DType::I64).unwrap().dtype(), DType::I64);
    assert_eq!(t1.to_dtype(DType::I32).unwrap().dtype(), DType::I32);
    assert_eq!(t1.to_dtype(DType::U32).unwrap().dtype(), DType::U32);
    assert_eq!(t1.to_dtype(DType::U8).unwrap().dtype(), DType::U8);

    let t1 = Tensor::new(&[0i32, 1, 2], device).unwrap();
    assert_eq!(t1.to_dtype(DType::F64).unwrap().dtype(), DType::F64);
    assert_eq!(t1.to_dtype(DType::F32).unwrap().dtype(), DType::F32);
    assert_eq!(t1.to_dtype(DType::I64).unwrap().dtype(), DType::I64);
    assert_eq!(t1.to_dtype(DType::U32).unwrap().dtype(), DType::U32);
    assert_eq!(t1.to_dtype(DType::U8).unwrap().dtype(), DType::U8);

    let t1 = Tensor::new(&[0i64, 1, 2], device).unwrap();
    assert_eq!(t1.to_dtype(DType::F64).unwrap().dtype(), DType::F64);
    assert_eq!(t1.to_dtype(DType::F32).unwrap().dtype(), DType::F32);
    assert_eq!(t1.to_dtype(DType::I32).unwrap().dtype(), DType::I32);
    assert_eq!(t1.to_dtype(DType::U32).unwrap().dtype(), DType::U32);
    assert_eq!(t1.to_dtype(DType::U8).unwrap().dtype(), DType::U8);

    let t1 = Tensor::new(&[0u32, 1, 2], device).unwrap();
    assert_eq!(t1.to_dtype(DType::F64).unwrap().dtype(), DType::F64);
    assert_eq!(t1.to_dtype(DType::F32).unwrap().dtype(), DType::F32);
    assert_eq!(t1.to_dtype(DType::I32).unwrap().dtype(), DType::I32);
    assert_eq!(t1.to_dtype(DType::I64).unwrap().dtype(), DType::I64);
    assert_eq!(t1.to_dtype(DType::U8).unwrap().dtype(), DType::U8);

    let t1 = Tensor::new(&[0u8, 1, 2], device).unwrap();
    assert_eq!(t1.to_dtype(DType::F64).unwrap().dtype(), DType::F64);
    assert_eq!(t1.to_dtype(DType::F32).unwrap().dtype(), DType::F32);
    assert_eq!(t1.to_dtype(DType::I32).unwrap().dtype(), DType::I32);
    assert_eq!(t1.to_dtype(DType::I64).unwrap().dtype(), DType::I64);
    assert_eq!(t1.to_dtype(DType::U32).unwrap().dtype(), DType::U32);
}
