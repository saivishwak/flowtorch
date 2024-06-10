use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    // Path to the output directory
    let out_dir = env::var("OUT_DIR").unwrap();

    // Inform Cargo about the dependency on the generated PTX files
    println!("cargo:rerun-if-changed=src");

    // Compile CUDA files and generate PTX files in the output directory
    let src_dir = Path::new("src");
    let cu_files = fs::read_dir(src_dir)
        .expect("Failed to read src directory")
        .filter_map(|entry| {
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();
            if path.is_file() && path.extension().unwrap_or_default() == "cu" {
                Some(path)
            } else {
                None
            }
        });

    for cu_file in cu_files {
        let cu_file_name = cu_file.file_name().unwrap().to_str().unwrap();
        let ptx_file_name = format!("{}.ptx", cu_file.file_stem().unwrap().to_str().unwrap());
        let ptx_output_path = Path::new(&out_dir).join(&ptx_file_name);

        let output = Command::new("nvcc")
            .arg("-ptx")
            .arg("-o")
            .arg(&ptx_output_path)
            .arg(&cu_file)
            .output();

        match output {
            Ok(_) => println!("{} compiled successfully!", cu_file_name),
            Err(e) => println!("Failed to compile {}: {}", cu_file_name, e),
        }
    }
}
