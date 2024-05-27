use std::fmt::Display;

use crate::Tensor;

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let formatted_string = self.as_string(Some(false)); // Auto detect trucation;
        match formatted_string {
            Ok(string) => {
                return write!(
                    f,
                    "Tensor({}, dtype; {}, device; {})",
                    string,
                    self.dtype(),
                    self.device()
                );
            }
            Err(e) => {
                return write!(f, "{}", e);
            }
        }
    }
}
