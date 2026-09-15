import numpy as np
import msgpack
import msgpack_numpy as m
import time

raw_unpackb = msgpack.unpackb
m.patch()


def bench_serialize(value, iterations):
    start_time = time.perf_counter()
    for _ in range(iterations):
        msgpack.packb(value)
    total_time = time.perf_counter() - start_time

    return total_time / iterations * 1000


def bench_deserialize_view(value, iterations):
    packed = msgpack.packb(value)

    start_time = time.perf_counter()
    for _ in range(iterations):
        msgpack.unpackb(packed)
    total_time = time.perf_counter() - start_time

    return total_time / iterations * 1000


def bench_deserialize_owned(value, iterations):
    packed = msgpack.packb(value)

    start_time = time.perf_counter()
    for _ in range(iterations):
        [array.copy() for array in msgpack.unpackb(packed)]
    total_time = time.perf_counter() - start_time

    return total_time / iterations * 1000


def bench_raw_unpack(value, iterations):
    packed = msgpack.packb(value)

    start_time = time.perf_counter()
    for _ in range(iterations):
        raw_unpackb(packed)
    total_time = time.perf_counter() - start_time

    return total_time / iterations * 1000


def bench_array_construction(value, iterations):
    packed = msgpack.packb(value)
    decoded = raw_unpackb(packed)

    start_time = time.perf_counter()
    for _ in range(iterations):
        [m.decode(item) for item in decoded]
    total_time = time.perf_counter() - start_time

    return total_time / iterations * 1000


def benchmark_case(dtype, array_size, array_count, iterations):
    array = np.arange(array_size, dtype=dtype)
    arrays = [array] * array_count
    print(
        f"{np.dtype(dtype).name},{array_size},{array_count},"
        f"{bench_serialize(arrays, iterations):.3f},"
        f"{bench_deserialize_view(arrays, iterations):.3f},"
        f"{bench_deserialize_owned(arrays, iterations):.3f},"
        f"{bench_raw_unpack(arrays, iterations):.3f},"
        f"{bench_array_construction(arrays, iterations):.3f}"
    )


def benchmark_large_array(dtype, iterations):
    array = np.zeros((1024, 2048), dtype=dtype)
    arrays = [array] * 10
    print(
        f"{np.dtype(dtype).name},"
        f"{bench_serialize(arrays, iterations):.3f},"
        f"{bench_deserialize_view(arrays, iterations):.3f},"
        f"{bench_deserialize_owned(arrays, iterations):.3f},"
        f"{bench_raw_unpack(arrays, iterations):.3f},"
        f"{bench_array_construction(arrays, iterations):.3f}"
    )


if __name__ == "__main__":
    iterations = 10

    print(
        "dtype,array_size,array_count,serialize_ms,deserialize_view_ms,"
        "deserialize_owned_ms,raw_unpack_ms,array_construction_ms"
    )
    for dtype in (np.float32, np.float16):
        benchmark_case(dtype, 1000, 10000, iterations)
        benchmark_case(dtype, 100, 100000, iterations)

    print("large-array benchmark: shape=(1024, 2048), array_count=10")
    print(
        "dtype,serialize_ms,deserialize_view_ms,deserialize_owned_ms,"
        "raw_unpack_ms,array_construction_ms"
    )
    benchmark_large_array(np.float32, iterations)
    benchmark_large_array(np.float16, iterations)
