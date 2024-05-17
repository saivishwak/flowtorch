use flowtorch_core::{DType, Device, Tensor};

fn zeros(device: &Device) -> Result<(), ()> {
    let tensor = Tensor::zeros(&[5, 2], DType::F32, device)?;
    let shape = tensor.shape();
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], 2);
    assert_eq!(DType::F32, tensor.dtype());
    Ok(())
}

// fn ones(device: &Device) -> Result<()> {
//     assert_eq!(
//         Tensor::ones((2, 3), DType::U8, device)?.to_vec2::<u8>()?,
//         [[1, 1, 1], [1, 1, 1]],
//     );
//     assert_eq!(
//         Tensor::ones((2, 3), DType::U32, device)?.to_vec2::<u32>()?,
//         [[1, 1, 1], [1, 1, 1]],
//     );
//     assert_eq!(
//         Tensor::ones((2, 3), DType::I64, device)?.to_vec2::<i64>()?,
//         [[1, 1, 1], [1, 1, 1]],
//     );
//     assert_eq!(
//         Tensor::ones((2, 3), DType::F32, device)?.to_vec2::<f32>()?,
//         [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
//     );
//     assert_eq!(
//         Tensor::ones((2, 3), DType::F64, device)?.to_vec2::<f64>()?,
//         [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
//     );
//     Ok(())
// }

// fn full(device: &Device) -> Result<()> {
//     assert_eq!(
//         Tensor::full(42u32, (2, 3), device)?.to_vec2::<u32>()?,
//         [[42, 42, 42], [42, 42, 42]],
//     );
//     Ok(())
// }

#[test]
fn test_zeros() {
    let device = Device::Cpu;
    let _ = zeros(&device);
}
