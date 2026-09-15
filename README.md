# msgpack-numpy-rs

![Crates.io](https://img.shields.io/crates/v/msgpack-numpy)
![Docs.rs](https://docs.rs/msgpack-numpy/badge.svg)
![License](https://img.shields.io/crates/l/msgpack-numpy)

This crate does what Python's [msgpack-numpy](https://github.com/lebedov/msgpack-numpy/) does in Rust, and a lot [faster](#benchmarks). It serializes and deserializes NumPy scalars and arrays to and from the [MessagePack](https://msgpack.org/) format, in the same serialized formats as the Python counterpart, so they could interoperate with each other. It enables processing NumPy arrays in a different service in Rust through IPC, or saving Machine Learning results to disk (better paired with compression).

## Overview

- It supports `bool`, `u8`, `i8`, `u16`, `i16`, `f16` (through the [`half`](https://crates.io/crates/half) crate), `u32`, `i32`, `f32`, `u64`, `i64`, `f64`.
- No support for arrays with complex numbers (`'c'`), byte strings (`'S'`), unicode strings (`'U'`), or other non-primitive types as elements. No support for structured/tuple data types (`'V'`), or object-type data that need pickling (`'O'`) ([ref](https://github.com/lebedov/msgpack-numpy/blob/0.4.8/msgpack_numpy.py)).
- However, during deserialization, we allow unsupported types to be deserialized as the `Unsupported` variant. This ensures deserialization can continue and the supported portions of data can be used.
- Scalars and arrays are represented as separate types, each of which being an enum of different element type variants. They come with convenient conversion methods (backed by the [`num-traits`](https://crates.io/crates/num-traits) crate) to the desired target primitive types. Example: `f16`, `f32`, `f64` can all be converted to `f64`, or `f16` with loss. This allows flexibility during deserialization, without explicit pattern matching and conditional conversion. It would be similar to NumPy's `.astype(np.float64)` / `.astype(np.float16)`. Notably, `bool` is convertible to numeric types as `(0, 1)`, but not from numeric types using these methods. Of course, you can do your own conversion after matching with the `Bool` variant.
- Arrays use the [`ndarray`](https://crates.io/crates/ndarray) crate, and have dynamic shapes. This enables users to leverage Rust's numeric [ecosystem](https://docs.rs/ndarray/latest/ndarray/index.html#the-ndarray-ecosystem) for the deserialized arrays.
- Array handling using `CowNDArray` could be zero-copy when array buffers in the serialized slice have good alignment, although MessagePack doesn't guarantee this.
- It depends on [`serde`](https://crates.io/crates/serde). In addition, it makes sense to use a correct MessagePack implementation, such as [`rmp-serde`](https://crates.io/crates/rmp-serde), which is used in the examples below, although it doesn't need to be a dependency, due to `serde`'s design.
- Only little-endian targets are supported. The crate rejects big-endian targets at compile time because NumPy dtypes emitted by this implementation are explicitly little-endian.



## Motivation

There hasn't been consensus on a good format that is both flexible and efficient for serializing NumPy arrays. They are unique in that they are blocks of bytes in nature, but also have numeric types and shapes. Programmers working on Machine Learning problems found MessagePack to have interesting properties. It is compact with a [type system](https://github.com/msgpack/msgpack/blob/master/spec.md), and has a wide range of language support. The package [msgpack-numpy](https://github.com/lebedov/msgpack-numpy/) provides de-/serialization for NumPy arrays, standalone or enclosed in arbitrary organizational depths, to be sent over the network, or saved to disk, in a compact format.

If one looks for a more production-oriented, performant format, they might consider [Apache Arrow](https://arrow.apache.org/), [Parquet](https://parquet.apache.org/), or [Protocol Buffers](https://protobuf.dev/). However, these formats are not as flexible as MessagePack when you need to store intermediate Machine Learning results. In practice, MessagePack with Numpy array support can be quite a good choice for many of these use cases.

This Rust version aims to provide a faster alternative to the Python version, with the same serialized formats as the Python counterpart so they could interoperate with each other. You could use this as a building block for your own Machine Learning pipeline in Rust, or as a way to communicate between Python and Rust.

## Examples

```rust
use std::fs::File;
use std::io::Read;
use msgpack_numpy::NDArray;

fn main() {
    let filepath = "tests/data/ndarray_bool.msgpack";
    let mut file = File::open(filepath).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    let deserialized: NDArray = rmp_serde::from_slice(&buf).unwrap();

    match &deserialized {
        NDArray::Bool(array) => {
            println!("{:?}", array);
        }
        _ => panic!("Expected NDArray::Bool"),
    }

    // returns an Option, None if conversion is not possible
    let arr = deserialized.into_u8_array().unwrap();
    println!("{:?}", arr);
}
```

Please see more in `examples/`.

## Benchmarks

These single-threaded benchmarks were run on Ubuntu 22.04 with an Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz. Rust used release mode with rustc 1.98.0. Python used Python 3.10.12, NumPy 2.0.0, msgpack 1.0.8, and msgpack-numpy 0.4.8. Only in-memory array serialization and deserialization are measured. See `benches/` for the benchmark code.

This table applies to the owned `NDArray`. Python uses its default payload-backed view, matching what users experience without an explicit copy.


| Array Type | Array Size | Arrays | Operation   | Python (ms) | Rust (ms) | Speedup |
| ---------- | ---------- | ------ | ----------- | ----------- | --------- | ------- |
| f32        | 1000       | 10000  | Serialize   | 83.8        | 30.0      | 2.8x    |
|            |            |        | Deserialize | 41.4        | 36.7      | 1.1x    |
|            | 100        | 100000 | Serialize   | 262.9       | 44.6      | 5.9x    |
|            |            |        | Deserialize | 265.7       | 79.6      | 3.3x    |
| f16        | 1000       | 10000  | Serialize   | 30.6        | 11.1      | 2.8x    |
|            |            |        | Deserialize | 29.8        | 10.8      | 2.8x    |
|            | 100        | 100000 | Serialize   | 230.6       | 18.0      | 12.8x   |
|            |            |        | Deserialize | 247.6       | 45.2      | 5.5x    |


The Rust implementation is faster in these cases, with the largest improvements for many small arrays. Python's de-/serialization logic is written in C through NumPy, but small arrays reduce this benefit because each array is also a Python object. This range of array sizes is typical for Machine Learning use cases such as feature embeddings.

### Zero-Copy Deserialization (when Good Alignment)

An owned `NDArray` must create an independently owned, correctly aligned typed allocation. `CowNDArray` can instead borrow the serialized payload when its address has the required alignment, falling back to an owned allocation otherwise. MessagePack itself does not guarantee alignment.

Python's msgpack-numpy takes a related but not ownership-equivalent approach: `msgpack.unpackb` creates a Python `bytes` object for each array payload, then msgpack-numpy calls `np.ndarray(buffer=payload, ...)`. The resulting NumPy array reports `OWNDATA=False` and retains that `bytes` object through its `.base` reference, so Python's garbage collector keeps the payload alive. It does not retain the original packed input buffer. By contrast, aligned Rust `CowNDArray` data borrows directly from that input slice.

The following benchmark uses arrays with shape `(1024, 2048)`, 10 arrays per iteration.


| Data Type | Operation                  | Python (ms) | Rust (ms) | Speedup | Cow borrowed |
| --------- | -------------------------- | ----------- | --------- | ------- | ------------ |
| f32       | Serialize                  | 140.5       | 71.9      | 2.0x    | -            |
|           | Deserialize (`NDArray`)    | 17.3        | 77.6      | 0.2x    | -            |
|           | Deserialize (`CowNDArray`) | 17.3        | 43.7      | 0.4x    | 2/10         |
| f16       | Serialize                  | 50.5        | 37.2      | 1.4x    | -            |
|           | Deserialize (`NDArray`)    | 8.5         | 16.2      | 0.5x    | -            |
|           | Deserialize (`CowNDArray`) | 8.5         | 3.4       | 2.5x    | 5/10         |


Speedup is Python time divided by Rust time, so values below `1x` mean Python was faster. In an isolated decomposition of the large-array case, Python spent about 9.0 ms (`f16`) or 18.8 ms (`f32`) in raw MessagePack unpacking, while constructing all 10 NumPy views over the resulting payload objects took about 0.02 ms. These component timings are diagnostic rather than additive because the end-to-end decoder constructs arrays inside its object hook.

The measured borrow counts explain why Rust `CowNDArray` gains more for `f16` than `f32`. `CowNDArray` supports `rmp_serde::from_slice` (consuming a slice held in memory), but not `rmp_serde::from_read` (streaming from a reader), because borrowed array data must not outlive the serialized bytes.

### Default View vs Owned Copy

The following Cartesian comparison uses the same 10 `f32` arrays of shape `(1024, 2048)` as the benchmark above. Python's default view is not equivalent to `CowNDArray`: Python retains newly decoded payload objects and permits unaligned arrays, while `CowNDArray` borrows the original input only when aligned and otherwise copies.


| Python result | Python (ms) | Rust result  | Rust (ms) | Speedup | Ownership-equivalent |
| ------------- | ----------- | ------------ | --------- | ------- | -------------------- |
| Default view  | 17.3        | `NDArray`    | 77.6      | 0.2x    | No                   |
|               | 17.3        | `CowNDArray` | 43.7      | 0.4x    | No                   |
| Owned copy    | 193.0       | `NDArray`    | 77.6      | 2.5x    | Yes                  |
|               | 193.0       | `CowNDArray` | 43.7      | 4.4x    | No                   |


If you really want complete zero-copy deserialization, you should try some other format, like [Apache Arrow](https://arrow.apache.org/).

## Notes

### `Scalar` Type

There is not a good reason to serialize using `Scalar`, because you end up representing primitive types with a lot of metadata. This type exists for compatibility reasons - it helps deserialize scalars already serialized this way.

### Dependency on `ndarray`

This section is similar to [`numpy`](https://github.com/pyo3/rust-numpy)'s.

This crate uses types from `ndarray` in its public API. `ndarray` is re-exported in the crate root so that you do not need to specify it as a direct dependency.

Furthermore, this crate is compatible with multiple versions of `ndarray` and therefore depends on a range of semver-incompatible versions, currently `>=0.15, <0.18`. Cargo may resolve this crate's dependency to `0.17.1` even if you pin `ndarray` to `0.15.6` in your own project. This can result in two versions of `ndarray` and compilation errors like:

```text
     = note: `ArrayBase<CowRepr<'_, f32>, Dim<IxDynImpl>>` and `ArrayBase<CowRepr<'_, f32>, Dim<IxDynImpl>>` have similar names, but are actually distinct types
note: `ArrayBase<CowRepr<'_, f32>, Dim<IxDynImpl>>` is defined in crate `ndarray`
    --> /home/ubuntu/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.15.6/src/lib.rs:1268:1
     |
1268 | pub struct ArrayBase<S, D>
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^
note: `ArrayBase<CowRepr<'_, f32>, Dim<IxDynImpl>>` is defined in crate `ndarray`
    --> /home/ubuntu/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.17.1/src/lib.rs
     |
1280 | pub struct ArrayBase<S, D>
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^
     = note: perhaps two different versions of crate `ndarray` are being used?
```

It can therefore be necessary to manually unify these dependencies. For example, if you specify the following dependencies

```toml
msgpack-numpy = "0.1.3"
ndarray = "0.15.6"
```

this may depend on both version `0.15.6` and `0.17.1` of `ndarray` even though `0.15.6` is within the supported range. To unify them on `0.15.6`, run

```bash
cargo update --package ndarray:0.17.1 --precise 0.15.6
```

to achieve a single dependency on version `0.15.6` of `ndarray`. Check your lock file to verify that this worked.

## License

This project is licensed under the MIT license.