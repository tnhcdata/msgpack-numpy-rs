use ctor::ctor;
use half::f16;
use msgpack_numpy::{CowNDArray, NDArray, Scalar};
use ndarray::{Array1, arr1, arr2, arr3};
use rstest::rstest;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

const DATA_DIR: &str = "tests/data";

#[ctor(unsafe)]
fn setup() {
    println!("Running setup...");
    let status = std::process::Command::new("tests/venv/bin/python")
        .args(["tests/helpers/serialize.py"])
        .status()
        .unwrap();

    assert!(status.success(), "Setup failed to run successfully");
    println!("Setup completed.");
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

#[rstest]
#[case("scalar_bool.msgpack", Scalar::Bool(true))]
#[case("scalar_uint8.msgpack", Scalar::U8(255))]
#[case("scalar_int8.msgpack", Scalar::I8(-128))]
#[case("scalar_uint16.msgpack", Scalar::U16(65535))]
#[case("scalar_int16.msgpack", Scalar::I16(-32768))]
#[case("scalar_float16.msgpack", Scalar::F16(f16::from_f32(1.0)))]
#[case("scalar_uint32.msgpack", Scalar::U32(4294967295))]
#[case("scalar_int32.msgpack", Scalar::I32(-2147483648))]
#[case("scalar_float32.msgpack", Scalar::F32(1.0))]
#[case("scalar_uint64.msgpack", Scalar::U64(18446744073709551615))]
#[case("scalar_int64.msgpack", Scalar::I64(-9223372036854775808))]
#[case("scalar_float64.msgpack", Scalar::F64(1.0))]
// unsupported but can be deserialized by this crate
#[case("scalar_complex64.msgpack", Scalar::Unsupported)] // 'c'
#[case("scalar_bytestring.msgpack", Scalar::Unsupported)] // 'S'
#[case("scalar_unicode_string.msgpack", Scalar::Unsupported)] // 'U'
fn test_scalar_deserialization(#[case] filename: &str, #[case] expected: Scalar) {
    let filepath = format!("{DATA_DIR}/{filename}");
    let serialized = read_file(&filepath);
    let deserialized: Scalar = deserialize(&serialized);

    match (deserialized, expected) {
        (Scalar::Bool(a), Scalar::Bool(b)) => assert_eq!(a, b),
        (Scalar::U8(a), Scalar::U8(b)) => assert_eq!(a, b),
        (Scalar::I8(a), Scalar::I8(b)) => assert_eq!(a, b),
        (Scalar::U16(a), Scalar::U16(b)) => assert_eq!(a, b),
        (Scalar::I16(a), Scalar::I16(b)) => assert_eq!(a, b),
        (Scalar::F16(a), Scalar::F16(b)) => assert_eq!(a, b),
        (Scalar::U32(a), Scalar::U32(b)) => assert_eq!(a, b),
        (Scalar::I32(a), Scalar::I32(b)) => assert_eq!(a, b),
        (Scalar::F32(a), Scalar::F32(b)) => assert_eq!(a, b),
        (Scalar::U64(a), Scalar::U64(b)) => assert_eq!(a, b),
        (Scalar::I64(a), Scalar::I64(b)) => assert_eq!(a, b),
        (Scalar::F64(a), Scalar::F64(b)) => assert_eq!(a, b),
        (Scalar::Unsupported, Scalar::Unsupported) => (),
        _ => panic!("Mismatched types for {}", filename),
    }
}

fn assert_float_eq<T>(a: T, b: T)
where
    T: num_traits::Float + std::fmt::Debug,
{
    if a.is_nan() && b.is_nan() {
        return; // Both are NaN, consider them equal
    }
    if a.is_infinite() && b.is_infinite() {
        assert_eq!(
            a.signum(),
            b.signum(),
            "Infinite values have different signs"
        );
        return;
    }
    assert_eq!(a, b);
}

#[rstest]
// Boolean
#[case("ndarray_bool.msgpack", NDArray::Bool(arr1(&[true, false, true, true, false]).into_dyn()))]
// Unsigned integers
#[case("ndarray_uint8.msgpack", NDArray::U8(arr1(&[0, 1, 255, 128, 254]).into_dyn()))]
#[case("ndarray_uint16.msgpack", NDArray::U16(arr1(&[0, 1, 65535, 32768, 65534]).into_dyn()))]
#[case("ndarray_uint32.msgpack", NDArray::U32(arr1(&[0, 1, 4294967295, 2147483648, 4294967294]).into_dyn()))]
#[case("ndarray_uint64.msgpack", NDArray::U64(arr1(&[0, 1, 18446744073709551615, 9223372036854775808, 18446744073709551614]).into_dyn()))]
// Signed integers
#[case("ndarray_int8.msgpack", NDArray::I8(arr1(&[-128, -1, 0, 1, 127]).into_dyn()))]
#[case("ndarray_int16.msgpack", NDArray::I16(arr1(&[-32768, -1, 0, 1, 32767]).into_dyn()))]
#[case("ndarray_int32.msgpack", NDArray::I32(arr1(&[-2147483648, -1, 0, 1, 2147483647]).into_dyn()))]
#[case("ndarray_int64.msgpack", NDArray::I64(arr1(&[-9223372036854775808, -1, 0, 1, 9223372036854775807]).into_dyn()))]
// Floating point numbers
#[case("ndarray_float16.msgpack", NDArray::F16(arr1(&[f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(-1.0), f16::from_f32(65504.0), f16::from_f32(-65504.0)]).into_dyn()))]
#[case("ndarray_float32.msgpack", NDArray::F32(arr1(&[0.0, 1.0, -1.0, f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]).into_dyn()))]
#[case("ndarray_float64.msgpack", NDArray::F64(arr1(&[0.0, 1.0, -1.0, f64::MAX, f64::MIN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]).into_dyn()))]
// Multidimensional arrays
#[case("ndarray_2d_int32.msgpack", NDArray::I32(arr2(&[[1, 2], [3, 4], [5, 6]]).into_dyn()))]
#[case("ndarray_2d_float32.msgpack", NDArray::F32(arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).into_dyn()))]
// others
#[case("ndarray_large_i32s.msgpack", NDArray::I32(Array1::from_iter(0..1000000).into_dyn()))]
#[case("ndarray_repeating_i32s.msgpack", NDArray::I32(Array1::from_iter((0..10).cycle().take(10000)).into_dyn()))]
// unsupported but can be deserialized by this crate
#[case("ndarray_complex64.msgpack", NDArray::Unsupported)] // 'c'
#[case("ndarray_bytestring.msgpack", NDArray::Unsupported)] // 'S'
#[case("ndarray_unicode_string.msgpack", NDArray::Unsupported)] // 'U'
#[case("ndarray_object.msgpack", NDArray::Unsupported)] // 'O'
#[case("ndarray_tuple_int32.msgpack", NDArray::Unsupported)] // 'V'
fn test_ndarray_deserialization(#[case] filename: &str, #[case] expected: NDArray) {
    let filepath = format!("{DATA_DIR}/{filename}");
    let serialized = read_file(&filepath);
    let deserialized: NDArray = deserialize(&serialized);

    match (deserialized, expected) {
        (NDArray::Bool(a), NDArray::Bool(b)) => assert_eq!(a, b),
        (NDArray::U8(a), NDArray::U8(b)) => assert_eq!(a, b),
        (NDArray::U16(a), NDArray::U16(b)) => assert_eq!(a, b),
        (NDArray::U32(a), NDArray::U32(b)) => assert_eq!(a, b),
        (NDArray::U64(a), NDArray::U64(b)) => assert_eq!(a, b),
        (NDArray::I8(a), NDArray::I8(b)) => assert_eq!(a, b),
        (NDArray::I16(a), NDArray::I16(b)) => assert_eq!(a, b),
        (NDArray::I32(a), NDArray::I32(b)) => assert_eq!(a, b),
        (NDArray::I64(a), NDArray::I64(b)) => assert_eq!(a, b),
        (NDArray::F16(a), NDArray::F16(b)) => {
            assert_eq!(a.shape(), b.shape());
            a.iter().zip(b.iter()).for_each(|(x, y)| {
                assert_float_eq(x.to_f32(), y.to_f32());
            });
        }
        (NDArray::F32(a), NDArray::F32(b)) => {
            assert_eq!(a.shape(), b.shape());
            a.iter().zip(b.iter()).for_each(|(x, y)| {
                assert_float_eq(*x, *y);
            });
        }
        (NDArray::F64(a), NDArray::F64(b)) => {
            assert_eq!(a.shape(), b.shape());
            a.iter().zip(b.iter()).for_each(|(x, y)| {
                assert_float_eq(*x, *y);
            });
        }
        (NDArray::Unsupported, NDArray::Unsupported) => (),
        _ => panic!("Mismatched types for {}", filename),
    }
}

#[rstest]
// Boolean
#[case("ndarray_bool.msgpack", CowNDArray::Bool(arr1(&[true, false, true, true, false]).into_dyn().into()))]
// Unsigned integers
#[case("ndarray_uint8.msgpack", CowNDArray::U8(arr1(&[0, 1, 255, 128, 254]).into_dyn().into()))]
#[case("ndarray_uint16.msgpack", CowNDArray::U16(arr1(&[0, 1, 65535, 32768, 65534]).into_dyn().into()))]
#[case("ndarray_uint32.msgpack", CowNDArray::U32(arr1(&[0, 1, 4294967295, 2147483648, 4294967294]).into_dyn().into()))]
#[case("ndarray_uint64.msgpack", CowNDArray::U64(arr1(&[0, 1, 18446744073709551615, 9223372036854775808, 18446744073709551614]).into_dyn().into()))]
// Signed integers
#[case("ndarray_int8.msgpack", CowNDArray::I8(arr1(&[-128, -1, 0, 1, 127]).into_dyn().into()))]
#[case("ndarray_int16.msgpack", CowNDArray::I16(arr1(&[-32768, -1, 0, 1, 32767]).into_dyn().into()))]
#[case("ndarray_int32.msgpack", CowNDArray::I32(arr1(&[-2147483648, -1, 0, 1, 2147483647]).into_dyn().into()))]
#[case("ndarray_int64.msgpack", CowNDArray::I64(arr1(&[-9223372036854775808, -1, 0, 1, 9223372036854775807]).into_dyn().into()))]
// Floating point numbers
#[case("ndarray_float16.msgpack", CowNDArray::F16(arr1(&[f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(-1.0), f16::from_f32(65504.0), f16::from_f32(-65504.0)]).into_dyn().into()))]
#[case("ndarray_float32.msgpack", CowNDArray::F32(arr1(&[0.0, 1.0, -1.0, f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]).into_dyn().into()))]
#[case("ndarray_float64.msgpack", CowNDArray::F64(arr1(&[0.0, 1.0, -1.0, f64::MAX, f64::MIN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]).into_dyn().into()))]
// Multidimensional arrays
#[case("ndarray_2d_int32.msgpack", CowNDArray::I32(arr2(&[[1, 2], [3, 4], [5, 6]]).into_dyn().into()))]
#[case("ndarray_2d_float32.msgpack", CowNDArray::F32(arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).into_dyn().into()))]
// others
#[case("ndarray_large_i32s.msgpack", CowNDArray::I32(Array1::from_iter(0..1000000).into_dyn().into()))]
#[case("ndarray_repeating_i32s.msgpack", CowNDArray::I32(Array1::from_iter((0..10).cycle().take(10000)).into_dyn().into()))]
// unsupported but can be deserialized by this crate
#[case("ndarray_complex64.msgpack", CowNDArray::Unsupported)] // 'c'
#[case("ndarray_bytestring.msgpack", CowNDArray::Unsupported)] // 'S'
#[case("ndarray_unicode_string.msgpack", CowNDArray::Unsupported)] // 'U'
#[case("ndarray_object.msgpack", CowNDArray::Unsupported)] // 'O'
#[case("ndarray_tuple_int32.msgpack", CowNDArray::Unsupported)] // 'V'
fn test_cowndarray_deserialization(#[case] filename: &str, #[case] expected: CowNDArray) {
    let filepath = format!("{DATA_DIR}/{filename}");
    let serialized = read_file(&filepath);
    let deserialized = deserialize(&serialized);

    match (deserialized, expected) {
        (CowNDArray::Bool(a), CowNDArray::Bool(b)) => assert_eq!(a, b),
        (CowNDArray::U8(a), CowNDArray::U8(b)) => assert_eq!(a, b),
        (CowNDArray::U16(a), CowNDArray::U16(b)) => assert_eq!(a, b),
        (CowNDArray::U32(a), CowNDArray::U32(b)) => assert_eq!(a, b),
        (CowNDArray::U64(a), CowNDArray::U64(b)) => assert_eq!(a, b),
        (CowNDArray::I8(a), CowNDArray::I8(b)) => assert_eq!(a, b),
        (CowNDArray::I16(a), CowNDArray::I16(b)) => assert_eq!(a, b),
        (CowNDArray::I32(a), CowNDArray::I32(b)) => assert_eq!(a, b),
        (CowNDArray::I64(a), CowNDArray::I64(b)) => assert_eq!(a, b),
        (CowNDArray::F16(a), CowNDArray::F16(b)) => {
            assert_eq!(a.shape(), b.shape());
            a.iter().zip(b.iter()).for_each(|(x, y)| {
                assert_float_eq(x.to_f32(), y.to_f32());
            });
        }
        (CowNDArray::F32(a), CowNDArray::F32(b)) => {
            assert_eq!(a.shape(), b.shape());
            a.iter().zip(b.iter()).for_each(|(x, y)| {
                assert_float_eq(*x, *y);
            });
        }
        (CowNDArray::F64(a), CowNDArray::F64(b)) => {
            assert_eq!(a.shape(), b.shape());
            a.iter().zip(b.iter()).for_each(|(x, y)| {
                assert_float_eq(*x, *y);
            });
        }
        (CowNDArray::Unsupported, CowNDArray::Unsupported) => (),
        _ => panic!("Mismatched types for {}", filename),
    }
}
