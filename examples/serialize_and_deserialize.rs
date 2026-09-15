use half::f16;
use msgpack_numpy::{NDArray, Scalar};
use ndarray::Array;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;

fn serialize<T: Serialize>(value: &T, filepath: &str) {
    let mut file = File::create(filepath).unwrap();
    rmp_serde::encode::write_named(&mut file, value).unwrap();
}

fn read_file(filepath: &str) -> Vec<u8> {
    let mut file = File::open(filepath).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    buf
}

fn deserialize<'a, T: Deserialize<'a>>(buf: &'a [u8]) -> T {
    rmp_serde::from_slice(buf).unwrap()
}

fn main() {
    {
        let filepath = "tests/data/example_float32_scalar.msgpack";
        let scalar = 0.1_f32;
        serialize(&Scalar::F32(scalar), filepath);
        let serialized = read_file(filepath);
        let deserialized: Scalar = deserialize(&serialized);

        match &deserialized {
            Scalar::F32(scalar) => {
                println!("{}", scalar);
            }
            _ => panic!("Expected Scalar::F32"),
        }

        let scalar_f64 = deserialized.to_f64().unwrap();
        println!("{scalar_f64}");
    }

    {
        let filepath = "tests/data/example_float16_array.msgpack";
        let arr = Array::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5])
            .into_dyn()
            .mapv(f16::from_f32);
        serialize(&NDArray::F16(arr), filepath);
        let serialized = read_file(filepath);
        let deserialized: NDArray = deserialize(&serialized);

        match &deserialized {
            NDArray::F16(array) => {
                println!("{:?}", array);
            }
            _ => panic!("Expected NDArray::F16"),
        }

        let arr_f32 = deserialized.into_f32_array().unwrap();
        println!("{arr_f32:?}");
    }
}
