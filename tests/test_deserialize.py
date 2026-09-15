import os
import subprocess
import pytest
import numpy as np
import msgpack
import msgpack_numpy as m

m.patch()

DATA_DIR = "tests/data"


@pytest.fixture(scope="session", autouse=True)
def setup():
    """Set up test data before running tests."""
    print("Running setup...")

    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Run the Rust binary to generate test data
    result = subprocess.run(
        ["cargo", "run", "--locked", "--example", "test_helpers_serialize"]
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to setup: {result.stderr}")

    print("Setup completed.")


def deserialize(filepath):
    with open(filepath, "rb") as f:
        return msgpack.unpackb(f.read())


@pytest.mark.parametrize("filename, expected_value", [
    ("scalar_bool.msgpack", np.array([True], dtype=np.bool_)[0]),
    ("scalar_uint8.msgpack", np.array([255], dtype=np.uint8)[0]),
    ("scalar_int8.msgpack", np.array([-128], dtype=np.int8)[0]),
    ("scalar_uint16.msgpack", np.array([65535], dtype=np.uint16)[0]),
    ("scalar_int16.msgpack", np.array([-32768], dtype=np.int16)[0]),
    ("scalar_float16.msgpack", np.array([1.0], dtype=np.float16)[0]),
    ("scalar_uint32.msgpack", np.array([4294967295], dtype=np.uint32)[0]),
    ("scalar_int32.msgpack", np.array([-2147483648], dtype=np.int32)[0]),
    ("scalar_float32.msgpack", np.array([1.0], dtype=np.float32)[0]),
    ("scalar_uint64.msgpack", np.array([18446744073709551615], dtype=np.uint64)[0]),
    ("scalar_int64.msgpack", np.array([-9223372036854775808], dtype=np.int64)[0]),
    ("scalar_float64.msgpack", np.array([1.0], dtype=np.float64)[0]),
])
def test_scalar_deserialization(filename, expected_value):
    deserialized = deserialize(f"{DATA_DIR}/{filename}")

    assert deserialized.dtype == expected_value.dtype
    assert deserialized == expected_value


@pytest.mark.parametrize("filename, expected_array", [
    # Boolean
    ("ndarray_bool.msgpack", np.array([True, False, True, True, False], dtype=np.bool_)),

    # Unsigned integers
    ("ndarray_uint8.msgpack", np.array([0, 1, 255, 128, 254], dtype=np.uint8)),
    ("ndarray_uint16.msgpack", np.array([0, 1, 65535, 32768, 65534], dtype=np.uint16)),
    ("ndarray_uint32.msgpack", np.array([0, 1, 4294967295, 2147483648, 4294967294], dtype=np.uint32)),
    ("ndarray_uint64.msgpack", np.array([0, 1, 18446744073709551615, 9223372036854775808, 18446744073709551614], dtype=np.uint64)),

    # Signed integers
    ("ndarray_int8.msgpack", np.array([-128, -1, 0, 1, 127], dtype=np.int8)),
    ("ndarray_int16.msgpack", np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)),
    ("ndarray_int32.msgpack", np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.int32)),
    ("ndarray_int64.msgpack", np.array([-9223372036854775808, -1, 0, 1, 9223372036854775807], dtype=np.int64)),

    # Floating point numbers
    ("ndarray_float16.msgpack", np.array([0.0, 1.0, -1.0, 65504.0, -65504.0], dtype=np.float16)),
    ("ndarray_float32.msgpack", np.array([0.0, 1.0, -1.0, np.finfo(np.float32).max, np.finfo(np.float32).min, np.inf, -np.inf, np.nan], dtype=np.float32)),
    ("ndarray_float64.msgpack", np.array([0.0, 1.0, -1.0, np.finfo(np.float64).max, np.finfo(np.float64).min, np.inf, -np.inf, np.nan], dtype=np.float64)),

    # Multidimensional arrays
    ("ndarray_2d_int32.msgpack", np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)),
    ("ndarray_2d_float32.msgpack", np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)),

    # Non-standard Rust layouts must serialize in logical row-major order
    ("ndarray_transposed_i32.msgpack", np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int32)),
    ("ndarray_reversed_i32.msgpack", np.array([4, 3, 2, 1], dtype=np.int32)),

    # others
    ("ndarray_large_i32s.msgpack", np.arange(1000000, dtype=np.int32)),
    ("ndarray_repeating_i32s.msgpack", np.tile(np.arange(10, dtype=np.int32), 1000)),
])
def test_ndarray_deserialization(filename, expected_array):
    deserialized = deserialize(f"{DATA_DIR}/{filename}")

    assert isinstance(deserialized, np.ndarray), f"Deserialized object is not a numpy array for {filename}"
    assert deserialized.dtype == expected_array.dtype, f"Dtype mismatch for {filename}: expected {expected_array.dtype}, got {deserialized.dtype}"
    assert deserialized.shape == expected_array.shape, f"Shape mismatch for {filename}: expected {expected_array.shape}, got {deserialized.shape}"

    np.testing.assert_array_equal(deserialized, expected_array)
