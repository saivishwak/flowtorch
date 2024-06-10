use crate::{formatter::PrintOptions, Formatter, Tensor};
use std::fmt::Display;

impl Tensor {
    pub fn fmt(&self, options: Option<PrintOptions>) -> Result<String, String> {
        let print_options = match options {
            Some(opts) => opts,
            None => PrintOptions::new(),
        };
        Formatter::new(self.clone(), print_options).fmt()
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let formatted_string = self.fmt(None);
        match formatted_string {
            Ok(string) => {
                write!(
                    f,
                    "Tensor({}, dtype; {}, device; {})",
                    string,
                    self.dtype(),
                    self.device()
                )
            }
            Err(e) => {
                write!(f, "{}", e)
            }
        }
    }
}
