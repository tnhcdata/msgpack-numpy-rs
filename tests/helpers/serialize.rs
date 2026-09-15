use half::f16;
use msgpack_numpy::{NDArray, Scalar};
use ndarray::{Array1, arr1, arr2, arr3, s};
use serde::Serialize;
use std::fs::File;

const DATA_DIR: &str = "tests/data";

fn serialize<T: Serialize>(value: &T, filepath: &str) {
    let mut file = File::create(filepath).unwrap();
    rmp_serde::encode::write_named(&mut file, value).unwrap();
}

#[rustfmt::skip]
fn main() {
    std::fs::create_dir_all(DATA_DIR).unwrap();

    // Scalar
    let cases = vec![
        ("scalar_bool.msgpack", Scalar::Bool(true)),
        ("scalar_uint8.msgpack", Scalar::U8(255)),
        ("scalar_int8.msgpack", Scalar::I8(-128)),
        ("scalar_uint16.msgpack", Scalar::U16(65535)),
        ("scalar_int16.msgpack", Scalar::I16(-32768)),
        ("scalar_float16.msgpack", Scalar::F16(f16::from_f32(1.0))),
        ("scalar_uint32.msgpack", Scalar::U32(4294967295)),
        ("scalar_int32.msgpack", Scalar::I32(-2147483648)),
        ("scalar_float32.msgpack", Scalar::F32(1.0)),
        ("scalar_uint64.msgpack", Scalar::U64(18446744073709551615)),
        ("scalar_int64.msgpack", Scalar::I64(-9223372036854775808)),
        ("scalar_float64.msgpack", Scalar::F64(1.0)),
    ];

    for (filename, case) in cases.into_iter() {
        let dtype = match case {
            Scalar::Bool(_) => "bool",
            Scalar::U8(_) => "uint8",
            Scalar::U16(_) => "uint16",
            Scalar::U32(_) => "uint32",
            Scalar::U64(_) => "uint64",
            Scalar::I8(_) => "int8",
            Scalar::I16(_) => "int16",
            Scalar::I32(_) => "int32",
            Scalar::I64(_) => "int64",
            Scalar::F16(_) => "float16",
            Scalar::F32(_) => "float32",
            Scalar::F64(_) => "float64",
            Scalar::Unsupported => panic!("Unsupported scalar type"),
        };
        let filepath = format!("{DATA_DIR}/{filename}");
        serialize(&case, &filepath);
        println!("Serialized {} scalar to {}", dtype, filepath);
    }

    // NDArray
    let cases = vec![
        ("ndarray_bool.msgpack", NDArray::Bool(arr1(&[true, false, true, true, false]).into_dyn())),
        // Unsigned integers
        ("ndarray_uint8.msgpack", NDArray::U8(arr1(&[0, 1, 255, 128, 254]).into_dyn())),
        ("ndarray_uint16.msgpack", NDArray::U16(arr1(&[0, 1, 65535, 32768, 65534]).into_dyn())),
        ("ndarray_uint32.msgpack", NDArray::U32(arr1(&[0, 1, 4294967295, 2147483648, 4294967294]).into_dyn())),
        ("ndarray_uint64.msgpack", NDArray::U64(arr1(&[0, 1, 18446744073709551615, 9223372036854775808, 18446744073709551614]).into_dyn())),
        // Signed integers
        ("ndarray_int8.msgpack", NDArray::I8(arr1(&[-128, -1, 0, 1, 127]).into_dyn())),
        ("ndarray_int16.msgpack", NDArray::I16(arr1(&[-32768, -1, 0, 1, 32767]).into_dyn())),
        ("ndarray_int32.msgpack", NDArray::I32(arr1(&[-2147483648, -1, 0, 1, 2147483647]).into_dyn())),
        ("ndarray_int64.msgpack", NDArray::I64(arr1(&[-9223372036854775808, -1, 0, 1, 9223372036854775807]).into_dyn())),
        // Floating point numbers
        ("ndarray_float16.msgpack", NDArray::F16(arr1(&[f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(-1.0), f16::from_f32(65504.0), f16::from_f32(-65504.0)]).into_dyn())),
        ("ndarray_float32.msgpack", NDArray::F32(arr1(&[0.0, 1.0, -1.0, f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]).into_dyn())),
        ("ndarray_float64.msgpack", NDArray::F64(arr1(&[0.0, 1.0, -1.0, f64::MAX, f64::MIN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]).into_dyn())),
        // Multidimensional arrays
        ("ndarray_2d_int32.msgpack", NDArray::I32(arr2(&[[1, 2], [3, 4], [5, 6]]).into_dyn())),
        ("ndarray_2d_float32.msgpack", NDArray::F32(arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).into_dyn())),
        // Non-standard layouts
        ("ndarray_transposed_i32.msgpack", NDArray::I32(arr2(&[[1, 2, 3], [4, 5, 6]]).reversed_axes().into_dyn())),
        ("ndarray_reversed_i32.msgpack", NDArray::I32(Array1::from_vec(vec![1, 2, 3, 4]).slice_move(s![..;-1]).into_dyn())),
        // others
        ("ndarray_large_i32s.msgpack", NDArray::I32(Array1::from_iter(0..1000000).into_dyn())),
        ("ndarray_repeating_i32s.msgpack", NDArray::I32(Array1::from_iter((0..10).cycle().take(10000)).into_dyn())),
    ];

    for (filename, case) in cases.into_iter() {
        let dtype = match case {
            NDArray::Bool(_) => "bool",
            NDArray::U8(_) => "uint8",
            NDArray::U16(_) => "uint16",
            NDArray::U32(_) => "uint32",
            NDArray::U64(_) => "uint64",
            NDArray::I8(_) => "int8",
            NDArray::I16(_) => "int16",
            NDArray::I32(_) => "int32",
            NDArray::I64(_) => "int64",
            NDArray::F16(_) => "float16",
            NDArray::F32(_) => "float32",
            NDArray::F64(_) => "float64",
            _ => panic!("Unexpected NDArray type"),
        };
        let filepath = format!("{DATA_DIR}/{filename}");
        serialize(&case, &filepath);
        println!("Serialized {} array to {}", dtype, filepath);
    }
}
