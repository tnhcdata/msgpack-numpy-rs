use crate::core::{CowNDArray, NDArray, Scalar};
use half::f16;
use serde::de::{self, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_bytes::{ByteBuf, Bytes};
use std::borrow::Cow;
use std::fmt;

// NumPy dtypes emitted by this crate are explicitly little-endian. The zero-copy
// byte casts below are therefore valid only when native byte order is little-endian.
#[cfg(not(target_endian = "little"))]
compile_error!("msgpack-numpy currently supports only little-endian targets");

// DType

enum DType {
    String(String),
    #[allow(dead_code)]
    Array(Vec<(String, String)>),
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DTypeVisitor;

        impl<'de> Visitor<'de> for DTypeVisitor {
            type Value = DType;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string or an array of tuples")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DType::String(value.to_string()))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut vec = Vec::new();
                while let Some((name, dtype)) = seq.next_element()? {
                    vec.push((name, dtype));
                }
                Ok(DType::Array(vec))
            }
        }

        deserializer.deserialize_any(DTypeVisitor)
    }
}

/***********************************************************************************************/
// Scalar

// impl Deserialize

impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ScalarVisitor;

        impl<'de> Visitor<'de> for ScalarVisitor {
            type Value = Scalar;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a numpy scaler in msgpack format")
            }

            // additional compatibility in case msgpack-python short-circuits during serialization
            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::Bool(v))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::I64(v))
            }

            // msgpack-python indeed short-circuits this during serialization
            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::F64(v))
            }

            // for NumPy's 'U' type
            fn visit_str<E>(self, _v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::Unsupported)
            }

            // for NumPy's 'S' type
            fn visit_bytes<E>(self, _v: &[u8]) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Scalar::Unsupported)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut nd: Option<bool> = None;
                let mut numpy_dtype: Option<DType> = None;
                let mut data: Option<ByteBuf> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "nd" => nd = Some(map.next_value()?),
                        "type" => numpy_dtype = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => return Err(de::Error::unknown_field(key, &["nd", "type", "data"])),
                    }
                }

                let nd = nd.ok_or_else(|| de::Error::missing_field("nd"))?;
                let numpy_dtype = numpy_dtype.ok_or_else(|| de::Error::missing_field("type"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                if nd {
                    return Err(de::Error::custom("nd should be false for numpy scalars"));
                }

                // we only support primitive numeric types for now

                match numpy_dtype {
                    DType::String(dtype) => {
                        match dtype.as_str() {
                            // convert through u8 to conform to NumPy's serialization behavior of booleans
                            "|b1" => TryInto::<[u8; 1]>::try_into(data.into_vec())
                                .map(|bytes| Scalar::Bool(bytes[0] != 0))
                                .map_err(|_| de::Error::custom("Invalid data for bool")),
                            "|u1" => TryInto::<[u8; 1]>::try_into(data.into_vec())
                                .map(|bytes| Scalar::U8(bytes[0]))
                                .map_err(|_| de::Error::custom("Invalid data for u8")),
                            "|i1" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I8(i8::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i8")),
                            "<u2" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::U16(u16::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for u16")),
                            "<i2" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I16(i16::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i16")),
                            "<f2" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::F16(f16::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for f16")),
                            "<u4" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::U32(u32::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for u32")),
                            "<i4" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I32(i32::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i32")),
                            "<f4" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::F32(f32::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for f32")),
                            "<u8" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::U64(u64::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for u64")),
                            "<i8" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::I64(i64::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for i64")),
                            "<f8" => data
                                .into_vec()
                                .try_into()
                                .map(|bytes| Scalar::F64(f64::from_le_bytes(bytes)))
                                .map_err(|_| de::Error::custom("Invalid data for f64")),
                            _ => Ok(Scalar::Unsupported),
                        }
                    }
                    DType::Array(_) => Ok(Scalar::Unsupported),
                }
            }
        }

        deserializer.deserialize_map(ScalarVisitor)
    }
}

// impl Serialize

impl Serialize for Scalar {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(3))?;

        state.serialize_entry(Bytes::new(b"nd"), &false)?;

        match self {
            // convert through u8 to conform to NumPy's serialization behavior of booleans
            Scalar::Bool(val) => serialize_value(&mut state, "|b1", &[*val as u8]),
            Scalar::U8(val) => serialize_value(&mut state, "|u1", &[*val]),
            Scalar::I8(val) => serialize_value(&mut state, "|i1", &val.to_le_bytes()),
            Scalar::U16(val) => serialize_value(&mut state, "<u2", &val.to_le_bytes()),
            Scalar::I16(val) => serialize_value(&mut state, "<i2", &val.to_le_bytes()),
            Scalar::F16(val) => serialize_value(&mut state, "<f2", &val.to_le_bytes()),
            Scalar::U32(val) => serialize_value(&mut state, "<u4", &val.to_le_bytes()),
            Scalar::I32(val) => serialize_value(&mut state, "<i4", &val.to_le_bytes()),
            Scalar::F32(val) => serialize_value(&mut state, "<f4", &val.to_le_bytes()),
            Scalar::U64(val) => serialize_value(&mut state, "<u8", &val.to_le_bytes()),
            Scalar::I64(val) => serialize_value(&mut state, "<i8", &val.to_le_bytes()),
            Scalar::F64(val) => serialize_value(&mut state, "<f8", &val.to_le_bytes()),
            Scalar::Unsupported => {
                return Err(serde::ser::Error::custom("Unsupported numpy dtype"));
            }
        }?;

        state.end()
    }
}

fn serialize_value<S>(state: &mut S, type_str: &str, val: &[u8]) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    state.serialize_entry(Bytes::new(b"type"), type_str)?;
    state.serialize_entry(Bytes::new(b"data"), Bytes::new(val))
}

/***********************************************************************************************/
// NDArray

use ndarray::{Array, ArrayBase, IxDyn};
use std::mem;

#[derive(thiserror::Error, Debug)]
enum NDArrayError {
    #[error("InvalidDataLength: {0}")]
    InvalidDataLength(String),

    #[error("ArrayShapeError: {0}")]
    ArrayShapeError(ndarray::ShapeError),
}

// impl Deserialize

impl<'de> Deserialize<'de> for NDArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NDArrayVisitor;

        impl<'de> Visitor<'de> for NDArrayVisitor {
            type Value = NDArray;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a numpy array in msgpack format")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut nd: Option<bool> = None;
                let mut numpy_dtype: Option<DType> = None;
                let mut kind: Option<ByteBuf> = None;
                let mut shape: Option<Vec<usize>> = None;
                let mut data: Option<ByteBuf> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "nd" => nd = Some(map.next_value()?),
                        "type" => numpy_dtype = Some(map.next_value()?),
                        "kind" => kind = Some(map.next_value()?),
                        "shape" => shape = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => {
                            return Err(de::Error::unknown_field(
                                key,
                                &["nd", "type", "kind", "shape", "data"],
                            ));
                        }
                    }
                }

                let nd = nd.ok_or_else(|| de::Error::missing_field("nd"))?;
                let numpy_dtype = numpy_dtype.ok_or_else(|| de::Error::missing_field("type"))?;
                let _kind = kind.ok_or_else(|| de::Error::missing_field("kind"))?;
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                if !nd {
                    return Err(de::Error::custom("nd should be true for numpy arrays"));
                }

                let shape = IxDyn(&shape);

                // we only support primitive numeric types for now

                match numpy_dtype {
                    DType::String(dtype) => {
                        match dtype.as_str() {
                            // convert through u8 to conform to NumPy's serialization behavior of booleans
                            "|b1" => Array::from_shape_vec(
                                shape,
                                data.into_iter().map(|v| v != 0).collect(),
                            )
                            .map(NDArray::Bool)
                            .map_err(de::Error::custom),
                            "|u1" => Array::from_shape_vec(shape, data.into_vec())
                                .map(NDArray::U8)
                                .map_err(de::Error::custom),
                            "|i1" => create_ndarray_from_bytes::<i8>(data.into_vec(), shape)
                                .map(NDArray::I8)
                                .map_err(de::Error::custom),
                            "<u2" => create_ndarray_from_bytes::<u16>(data.into_vec(), shape)
                                .map(NDArray::U16)
                                .map_err(de::Error::custom),
                            "<i2" => create_ndarray_from_bytes::<i16>(data.into_vec(), shape)
                                .map(NDArray::I16)
                                .map_err(de::Error::custom),
                            "<f2" => create_ndarray_from_bytes::<f16>(data.into_vec(), shape)
                                .map(NDArray::F16)
                                .map_err(de::Error::custom),
                            "<u4" => create_ndarray_from_bytes::<u32>(data.into_vec(), shape)
                                .map(NDArray::U32)
                                .map_err(de::Error::custom),
                            "<i4" => create_ndarray_from_bytes::<i32>(data.into_vec(), shape)
                                .map(NDArray::I32)
                                .map_err(de::Error::custom),
                            "<f4" => create_ndarray_from_bytes::<f32>(data.into_vec(), shape)
                                .map(NDArray::F32)
                                .map_err(de::Error::custom),
                            "<u8" => create_ndarray_from_bytes::<u64>(data.into_vec(), shape)
                                .map(NDArray::U64)
                                .map_err(de::Error::custom),
                            "<i8" => create_ndarray_from_bytes::<i64>(data.into_vec(), shape)
                                .map(NDArray::I64)
                                .map_err(de::Error::custom),
                            "<f8" => create_ndarray_from_bytes::<f64>(data.into_vec(), shape)
                                .map(NDArray::F64)
                                .map_err(de::Error::custom),
                            _ => Ok(NDArray::Unsupported),
                        }
                    }
                    DType::Array(_) => Ok(NDArray::Unsupported),
                }
            }
        }

        deserializer.deserialize_map(NDArrayVisitor)
    }
}

/// Creates an n-dimensional array from little-endian byte data.
///
/// # Type Parameters
///
/// * `T`: The target numeric type (e.g., f32, i64).
///
/// # Arguments
///
/// * `data`: Raw bytes to be decoded and reshaped.
/// * `shape`: The desired shape of the output array.
///
/// # Returns
///
/// An n-dimensional array of type `T` with the specified shape, or an error.
///
/// # Errors
///
/// Returns an error if:
/// * Data length isn't a multiple of `size_of::<T>()`.
/// * Specified shape doesn't match the decoded data length.
fn create_ndarray_from_bytes<T: bytemuck::Pod>(
    data: Vec<u8>,
    shape: IxDyn,
) -> Result<Array<T, IxDyn>, NDArrayError> {
    let values = match bytemuck::allocation::try_cast_vec(data) {
        // This zero-copy takeover is possible only when u8 and T have exactly
        // the same allocation layout (for the supported types, T = i8).
        Ok(values) => values,
        Err((_, data)) => bytes_to_vec(&data).ok_or_else(|| {
            NDArrayError::InvalidDataLength(format!(
                "Invalid data length for {} decoding",
                std::any::type_name::<T>()
            ))
        })?,
    };

    Array::from_shape_vec(shape, values).map_err(NDArrayError::ArrayShapeError)
}

fn bytes_to_vec<T: bytemuck::Pod>(data: &[u8]) -> Option<Vec<T>> {
    let size_of_t = mem::size_of::<T>();
    if size_of_t == 0 {
        return None;
    }
    let chunks = data.chunks_exact(size_of_t);
    if !chunks.remainder().is_empty() {
        return None;
    }

    // The crate has a compile-time little-endian guard, so native Pod reads
    // match the explicitly little-endian NumPy dtypes.
    Some(chunks.map(bytemuck::pod_read_unaligned).collect())
}

// impl Serialize

impl Serialize for NDArray {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(5))?;

        state.serialize_entry(Bytes::new(b"nd"), &true)?;

        match self {
            // convert through u8 to conform to NumPy's serialization behavior of booleans
            NDArray::Bool(arr) => serialize_ndarray(&mut state, "|b1", &arr.mapv(|v| v as u8)),
            NDArray::U8(arr) => serialize_ndarray(&mut state, "|u1", arr),
            NDArray::I8(arr) => serialize_ndarray(&mut state, "|i1", arr),
            NDArray::U16(arr) => serialize_ndarray(&mut state, "<u2", arr),
            NDArray::I16(arr) => serialize_ndarray(&mut state, "<i2", arr),
            NDArray::F16(arr) => serialize_ndarray(&mut state, "<f2", arr),
            NDArray::U32(arr) => serialize_ndarray(&mut state, "<u4", arr),
            NDArray::I32(arr) => serialize_ndarray(&mut state, "<i4", arr),
            NDArray::F32(arr) => serialize_ndarray(&mut state, "<f4", arr),
            NDArray::U64(arr) => serialize_ndarray(&mut state, "<u8", arr),
            NDArray::I64(arr) => serialize_ndarray(&mut state, "<i8", arr),
            NDArray::F64(arr) => serialize_ndarray(&mut state, "<f8", arr),
            NDArray::Unsupported => {
                return Err(serde::ser::Error::custom("Unsupported numpy dtype"));
            }
        }?;

        state.end()
    }
}

fn serialize_ndarray<S, A, T>(
    state: &mut S,
    type_str: &str,
    arr: &ArrayBase<A, IxDyn>,
) -> Result<(), S::Error>
where
    S: SerializeMap,
    A: ndarray::Data<Elem = T>,
    T: bytemuck::Pod,
{
    state.serialize_entry(Bytes::new(b"type"), type_str)?;
    state.serialize_entry(Bytes::new(b"kind"), Bytes::new(b""))?;
    state.serialize_entry(Bytes::new(b"shape"), &arr.shape())?;

    let data = ndarray_to_bytes(arr);
    state.serialize_entry(Bytes::new(b"data"), Bytes::new(&data))
}

/// Returns borrowed bytes for standard-layout arrays and copies arrays with
/// non-standard strides into logical (row-major) order.
fn ndarray_to_bytes<A: ndarray::Data<Elem = T>, T: bytemuck::Pod>(
    arr: &ArrayBase<A, IxDyn>,
) -> Cow<'_, [u8]> {
    if let Some(slice) = arr.as_slice() {
        // The crate has a compile-time little-endian guard, and Pod excludes
        // padding and invalid bit patterns.
        return Cow::Borrowed(bytemuck::cast_slice(slice));
    }

    let mut data = Vec::with_capacity(arr.len() * mem::size_of::<T>());
    for value in arr.iter() {
        data.extend_from_slice(bytemuck::bytes_of(value));
    }
    Cow::Owned(data)
}

/***********************************************************************************************/
// CowNDArray

use ndarray::{ArrayView, CowArray};

// impl Deserialize

impl<'de: 'a, 'a> Deserialize<'de> for CowNDArray<'a> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NDArrayVisitor<'a>(std::marker::PhantomData<&'a ()>);

        impl<'de: 'a, 'a> Visitor<'de> for NDArrayVisitor<'a> {
            type Value = CowNDArray<'a>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a numpy array in msgpack format")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut nd: Option<bool> = None;
                let mut numpy_dtype: Option<DType> = None;
                let mut kind: Option<&'a Bytes> = None;
                let mut shape: Option<Vec<usize>> = None;
                let mut data: Option<&'a Bytes> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "nd" => nd = Some(map.next_value()?),
                        "type" => numpy_dtype = Some(map.next_value()?),
                        "kind" => kind = Some(map.next_value()?),
                        "shape" => shape = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => {
                            return Err(de::Error::unknown_field(
                                key,
                                &["nd", "type", "kind", "shape", "data"],
                            ));
                        }
                    }
                }

                let nd = nd.ok_or_else(|| de::Error::missing_field("nd"))?;
                let numpy_dtype = numpy_dtype.ok_or_else(|| de::Error::missing_field("type"))?;
                let _kind = kind.ok_or_else(|| de::Error::missing_field("kind"))?;
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                if !nd {
                    return Err(de::Error::custom("nd should be true for numpy arrays"));
                }

                let shape = IxDyn(&shape);

                // we only support primitive numeric types for now

                match numpy_dtype {
                    DType::String(dtype) => {
                        match dtype.as_str() {
                            // convert through u8 to conform to NumPy's serialization behavior of booleans
                            "|b1" => Array::from_shape_vec(
                                shape,
                                data.into_iter().map(|v| *v != 0).collect(),
                            )
                            .map(CowArray::from)
                            .map(CowNDArray::Bool)
                            .map_err(de::Error::custom),
                            "|u1" => ArrayView::from_shape(shape, data)
                                .map(CowArray::from)
                                .map(CowNDArray::U8)
                                .map_err(de::Error::custom),
                            "|i1" => create_cowndarray_from_bytes::<i8>(data, shape)
                                .map(CowNDArray::I8)
                                .map_err(de::Error::custom),
                            "<u2" => create_cowndarray_from_bytes::<u16>(data, shape)
                                .map(CowNDArray::U16)
                                .map_err(de::Error::custom),
                            "<i2" => create_cowndarray_from_bytes::<i16>(data, shape)
                                .map(CowNDArray::I16)
                                .map_err(de::Error::custom),
                            "<f2" => create_cowndarray_from_bytes::<f16>(data, shape)
                                .map(CowNDArray::F16)
                                .map_err(de::Error::custom),
                            "<u4" => create_cowndarray_from_bytes::<u32>(data, shape)
                                .map(CowNDArray::U32)
                                .map_err(de::Error::custom),
                            "<i4" => create_cowndarray_from_bytes::<i32>(data, shape)
                                .map(CowNDArray::I32)
                                .map_err(de::Error::custom),
                            "<f4" => create_cowndarray_from_bytes::<f32>(data, shape)
                                .map(CowNDArray::F32)
                                .map_err(de::Error::custom),
                            "<u8" => create_cowndarray_from_bytes::<u64>(data, shape)
                                .map(CowNDArray::U64)
                                .map_err(de::Error::custom),
                            "<i8" => create_cowndarray_from_bytes::<i64>(data, shape)
                                .map(CowNDArray::I64)
                                .map_err(de::Error::custom),
                            "<f8" => create_cowndarray_from_bytes::<f64>(data, shape)
                                .map(CowNDArray::F64)
                                .map_err(de::Error::custom),
                            _ => Ok(CowNDArray::Unsupported),
                        }
                    }
                    DType::Array(_) => Ok(CowNDArray::Unsupported),
                }
            }
        }

        deserializer.deserialize_map(NDArrayVisitor(std::marker::PhantomData))
    }
}

fn create_cowndarray_from_bytes<'a, T: bytemuck::Pod>(
    data: &'a [u8],
    shape: IxDyn,
) -> Result<CowArray<'a, T, IxDyn>, NDArrayError> {
    let values = bytes_to_cow(data).ok_or_else(|| {
        NDArrayError::InvalidDataLength(format!(
            "Invalid data length for {} decoding",
            std::any::type_name::<T>()
        ))
    })?;

    match values {
        Cow::Borrowed(slice) => ArrayView::from_shape(shape, slice).map(CowArray::from),
        Cow::Owned(vec) => Array::from_shape_vec(shape, vec).map(CowArray::from),
    }
    .map_err(NDArrayError::ArrayShapeError)
}

fn bytes_to_cow<T: bytemuck::Pod>(data: &[u8]) -> Option<Cow<'_, [T]>> {
    let size_of_t = mem::size_of::<T>();
    if size_of_t == 0 || !data.chunks_exact(size_of_t).remainder().is_empty() {
        return None;
    }

    // Borrow when the MessagePack payload happens to have T's alignment.
    // Otherwise allocate correctly aligned storage and perform unaligned reads.
    match bytemuck::try_cast_slice(data) {
        Ok(slice) => Some(Cow::Borrowed(slice)),
        Err(_) => bytes_to_vec(data).map(Cow::Owned),
    }
}

// impl Serialize

impl<'a> Serialize for CowNDArray<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(5))?;

        state.serialize_entry(Bytes::new(b"nd"), &true)?;

        match self {
            // convert through u8 to conform to NumPy's serialization behavior of booleans
            CowNDArray::Bool(arr) => serialize_ndarray(&mut state, "|b1", &arr.mapv(|v| v as u8)),
            CowNDArray::U8(arr) => serialize_ndarray(&mut state, "|u1", arr),
            CowNDArray::I8(arr) => serialize_ndarray(&mut state, "|i1", arr),
            CowNDArray::U16(arr) => serialize_ndarray(&mut state, "<u2", arr),
            CowNDArray::I16(arr) => serialize_ndarray(&mut state, "<i2", arr),
            CowNDArray::F16(arr) => serialize_ndarray(&mut state, "<f2", arr),
            CowNDArray::U32(arr) => serialize_ndarray(&mut state, "<u4", arr),
            CowNDArray::I32(arr) => serialize_ndarray(&mut state, "<i4", arr),
            CowNDArray::F32(arr) => serialize_ndarray(&mut state, "<f4", arr),
            CowNDArray::U64(arr) => serialize_ndarray(&mut state, "<u8", arr),
            CowNDArray::I64(arr) => serialize_ndarray(&mut state, "<i8", arr),
            CowNDArray::F64(arr) => serialize_ndarray(&mut state, "<f8", arr),
            CowNDArray::Unsupported => {
                return Err(serde::ser::Error::custom("Unsupported numpy dtype"));
            }
        }?;

        state.end()
    }
}

/*********************************************************************************/
// tests

#[cfg(test)]
mod tests {
    use super::{NDArrayError, create_cowndarray_from_bytes, create_ndarray_from_bytes};
    use crate::core::{CowNDArray, NDArray, Scalar};
    use half::f16;
    use ndarray::{Array, IxDyn, arr2, s};

    #[test]
    fn test_scalar_serialization() {
        let cases = vec![
            Scalar::Bool(true),
            Scalar::U8(255),
            Scalar::I8(-128),
            Scalar::U16(65535),
            Scalar::I16(-32768),
            Scalar::F16(f16::from_f32(1.0)),
            Scalar::U32(4294967295),
            Scalar::I32(-2147483648),
            Scalar::F32(1.0),
            Scalar::U64(18446744073709551615),
            Scalar::I64(-9223372036854775808),
            Scalar::F64(1.0),
        ];

        for scalar in cases {
            let serialized = rmp_serde::to_vec_named(&scalar).unwrap();
            let deserialized: Scalar = rmp_serde::from_slice(&serialized).unwrap();
            assert_eq!(deserialized, scalar);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn test_ndarray_serialization() {
        let cases = vec![
            NDArray::Bool(Array::from_vec(vec![true, false]).into_dyn()),
            NDArray::U8(Array::from_vec(vec![1, 2, 3]).into_dyn()),
            NDArray::I8(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::U16(Array::from_vec(vec![1, 2, 3]).into_dyn()),
            NDArray::I16(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::F16(Array::from_vec(vec![1.0, 2.0]).into_dyn().mapv(f16::from_f32)),
            NDArray::U32(Array::from_vec(vec![1, 2, 3]).into_dyn()),
            NDArray::I32(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::F32(Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn()),
            NDArray::U64(Array::from_vec(vec![1, 2]).into_dyn()),
            NDArray::I64(Array::from_vec(vec![-1, 0, 1]).into_dyn()),
            NDArray::F64(Array::from_vec(vec![1.0, 2.0]).into_dyn()),
        ];

        for ndarray in cases {
            let serialized = rmp_serde::to_vec_named(&ndarray).unwrap();
            let deserialized: NDArray = rmp_serde::from_slice(&serialized).unwrap();

            assert_eq!(deserialized, ndarray);
        }
    }

    #[test]
    fn test_non_standard_layout_serialization() {
        let transposed = arr2(&[[1_i32, 2, 3], [4, 5, 6]]).reversed_axes().into_dyn();
        assert!(!transposed.is_standard_layout());

        let reversed = Array::from_vec(vec![1_i32, 2, 3, 4])
            .slice_move(s![..;-1])
            .into_dyn();
        assert!(!reversed.is_standard_layout());

        for expected in [transposed, reversed] {
            let serialized = rmp_serde::to_vec_named(&NDArray::I32(expected.clone())).unwrap();
            let deserialized: NDArray = rmp_serde::from_slice(&serialized).unwrap();
            assert_eq!(deserialized, NDArray::I32(expected));
        }
    }

    #[test]
    fn test_i8_deserialization_reuses_byte_allocation() {
        let bytes = vec![0_u8, 127, 128, 255];
        let bytes_ptr = bytes.as_ptr();
        let array = create_ndarray_from_bytes::<i8>(bytes, IxDyn(&[4])).unwrap();

        assert_eq!(array.as_ptr().cast::<u8>(), bytes_ptr);
        assert_eq!(array.as_slice().unwrap(), &[0, 127, -128, -1]);
    }

    #[test]
    fn test_malformed_byte_lengths_are_rejected() {
        assert!(matches!(
            create_ndarray_from_bytes::<u32>(vec![0; 3], IxDyn(&[1])),
            Err(NDArrayError::InvalidDataLength(_))
        ));
        assert!(matches!(
            create_cowndarray_from_bytes::<u32>(&[0; 3], IxDyn(&[1])),
            Err(NDArrayError::InvalidDataLength(_))
        ));
    }

    #[test]
    fn test_cow_deserialization_borrows_aligned_data_and_copies_misaligned_data() {
        let aligned_values = [1_u32, 2];
        let aligned_bytes = bytemuck::cast_slice(&aligned_values);
        let borrowed = create_cowndarray_from_bytes::<u32>(aligned_bytes, IxDyn(&[2])).unwrap();

        assert!(borrowed.is_view());
        assert_eq!(borrowed.as_ptr(), aligned_values.as_ptr());

        let backing = [0x0403_0201_u32, 0x0807_0605, 0x0c0b_0a09];
        let backing_bytes = bytemuck::cast_slice(&backing);
        let misaligned_bytes = &backing_bytes[1..9];
        let copied = create_cowndarray_from_bytes::<u32>(misaligned_bytes, IxDyn(&[2])).unwrap();

        assert!(!copied.is_view());
        assert_ne!(copied.as_ptr().cast::<u8>(), misaligned_bytes.as_ptr());
        assert_eq!(copied.as_slice().unwrap(), &[0x0504_0302, 0x0908_0706]);
    }

    #[test]
    #[rustfmt::skip]
    fn test_cowndarray_serialization() {
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
        let cases = vec![
            CowNDArray::Bool(Array::from_vec(vec![true, false]).into_dyn().into()),
            CowNDArray::U8(Array::from_vec(vec![1, 2, 3]).into_dyn().into()),
            CowNDArray::I8(Array::from_vec(vec![-1, 0, 1]).into_dyn().into()),
            CowNDArray::U16(Array::from_vec(vec![1, 2, 3]).into_dyn().into()),
            CowNDArray::I16(Array::from_vec(vec![-1, 0, 1]).into_dyn().into()),
            CowNDArray::F16(Array::from_vec(vec![1.0, 2.0]).into_dyn().mapv(f16::from_f32).into()),
            CowNDArray::U32(Array::from_vec(vec![1, 2, 3]).into_dyn().into()),
            CowNDArray::I32(Array::from_vec(vec![-1, 0, 1]).into_dyn().into()),
            CowNDArray::F32(Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn().into()),
            CowNDArray::U64(Array::from_vec(vec![1, 2]).into_dyn().into()),
            CowNDArray::I64(Array::from_vec(vec![-1, 0, 1]).into_dyn().into()),
            CowNDArray::F64(Array::from_vec(vec![1.0, 2.0]).into_dyn().into()),
        ];

        for ndarray in cases {
            let serialized = rmp_serde::to_vec_named(&ndarray).unwrap();
            let deserialized: CowNDArray = rmp_serde::from_slice(&serialized).unwrap();

            match (deserialized, ndarray) {
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
                _ => panic!("Mismatched types"),
            }
        }
    }
}
