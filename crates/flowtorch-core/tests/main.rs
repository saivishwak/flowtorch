#![allow(unused_imports)]
use flowtorch_core::{Device, DeviceT};
use tensor::test_tensor;

mod tensor;

#[cfg(feature = "cuda")]
mod cuda;

// Common Tests that needs to be run with different devices
fn run_tests(device: &Device) {
    test_tensor(device);
}

#[test]
fn main() {
    let device: Device = Device::Cpu;
    println!("Testing with CPU");
    run_tests(&device);

    #[cfg(feature = "cuda")]
    {
        let device = Device::new(DeviceT::Cuda(0)).unwrap();
        println!("CUDA Feature enabled - Testing with CUDA");
        run_tests(&device);
    }
}
