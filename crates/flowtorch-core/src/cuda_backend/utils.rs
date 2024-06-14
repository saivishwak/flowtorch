use crate::dtype::WithDType;

pub fn get_kernel_name<T: WithDType>(op: &'static str) -> String {
    let dtype = T::dtype().as_str();
    format!("{op}_{dtype}")
}
