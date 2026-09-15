use half::f16;
use msgpack_numpy::{CowNDArray, NDArray};
use ndarray::{Array1, Array2};
use serde::Serialize;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench_serialize<T: Serialize>(value: &T, iterations: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(rmp_serde::to_vec_named(value).unwrap());
    }
    start.elapsed() / iterations
}

fn bench_ndarray_deserialize(value: &[NDArray], iterations: u32) -> Duration {
    let buf = rmp_serde::to_vec_named(value).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(rmp_serde::from_slice::<Vec<NDArray>>(&buf).unwrap());
    }
    start.elapsed() / iterations
}

fn bench_cow_deserialize(value: &[NDArray], iterations: u32) -> Duration {
    let buf = rmp_serde::to_vec_named(value).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(rmp_serde::from_slice::<Vec<CowNDArray<'_>>>(&buf).unwrap());
    }
    start.elapsed() / iterations
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn benchmark_case(
    dtype: &str,
    array_size: usize,
    array_count: usize,
    array: NDArray,
    iterations: u32,
) {
    let arrays = vec![array; array_count];
    println!(
        "{dtype},{array_size},{array_count},{:.3},{:.3}",
        milliseconds(bench_serialize(&arrays, iterations)),
        milliseconds(bench_ndarray_deserialize(&arrays, iterations)),
    );
}

fn benchmark_zero_copy(dtype: &str, array: NDArray, iterations: u32) {
    let arrays = vec![array; 10];
    let buf = rmp_serde::to_vec_named(&arrays).unwrap();
    let cow_arrays: Vec<CowNDArray<'_>> = rmp_serde::from_slice(&buf).unwrap();
    let borrowed_arrays = cow_arrays
        .iter()
        .filter(|array| match array {
            CowNDArray::F16(array) => array.is_view(),
            CowNDArray::F32(array) => array.is_view(),
            _ => false,
        })
        .count();

    println!(
        "{dtype},{:.3},{:.3},{:.3},{borrowed_arrays}/{}",
        milliseconds(bench_serialize(&arrays, iterations)),
        milliseconds(bench_ndarray_deserialize(&arrays, iterations)),
        milliseconds(bench_cow_deserialize(&arrays, iterations)),
        arrays.len(),
    );
}

fn main() {
    let iterations = 10;

    println!("dtype,array_size,array_count,serialize_ms,ndarray_deserialize_ms");
    benchmark_case(
        "f32",
        1000,
        10000,
        NDArray::F32(Array1::range(0., 1000., 1.).into_dyn()),
        iterations,
    );
    benchmark_case(
        "f32",
        100,
        100000,
        NDArray::F32(Array1::range(0., 100., 1.).into_dyn()),
        iterations,
    );
    benchmark_case(
        "f16",
        1000,
        10000,
        NDArray::F16(Array1::range(0., 1000., 1.).mapv(f16::from_f32).into_dyn()),
        iterations,
    );
    benchmark_case(
        "f16",
        100,
        100000,
        NDArray::F16(Array1::range(0., 100., 1.).mapv(f16::from_f32).into_dyn()),
        iterations,
    );

    println!("zero-copy benchmark: shape=(1024, 2048), array_count=10");
    println!("dtype,serialize_ms,ndarray_deserialize_ms,cow_deserialize_ms,cow_borrowed");
    benchmark_zero_copy(
        "f32",
        NDArray::F32(Array2::zeros((1024, 2048)).into_dyn()),
        iterations,
    );
    benchmark_zero_copy(
        "f16",
        NDArray::F16(Array2::from_elem((1024, 2048), f16::ZERO).into_dyn()),
        iterations,
    );
}
