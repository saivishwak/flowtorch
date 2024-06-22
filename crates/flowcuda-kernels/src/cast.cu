#include <math.h>
#include "utils.cuh"

template <typename T, typename S>
__device__ void cast(const size_t numel, const T *data, S *out, bool is_contiguous, size_t *layout, size_t num_dims)
{
    if (is_contiguous)
    {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)
        {
            out[i] = data[i];
        }
    }
    else
    {
        size_t *dims = layout;
        size_t *stride = dims + num_dims;
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)
        {
            unsigned strided_i = get_strided_index(i, num_dims, dims, stride);
            out[strided_i] = data[i];
        }
    }
}

#define CAST_OP(TYPENAME, FN_NAME, TO_TYPENAME)                                                                                                         \
    extern "C" __global__ void FN_NAME(const size_t numel, const TYPENAME *data, TO_TYPENAME *out, bool is_contiguous, size_t *layout, size_t num_dims) \
    {                                                                                                                                                   \
        cast<TYPENAME, TO_TYPENAME>(numel, data, out, is_contiguous, layout, num_dims);                                                                 \
    }

// DataType Casting

// F64
CAST_OP(double, cast_f64_f32, float);
CAST_OP(double, cast_f64_i32, int32_t);
CAST_OP(double, cast_f64_i64, int64_t);
CAST_OP(double, cast_f64_u32, u_int32_t);
CAST_OP(double, cast_f64_u8, u_int8_t);

// F32
CAST_OP(float, cast_f32_f64, double);
CAST_OP(float, cast_f32_i32, int32_t);
CAST_OP(float, cast_f32_i64, int64_t);
CAST_OP(float, cast_f32_u32, u_int32_t);
CAST_OP(float, cast_f32_u8, u_int8_t);

// I32
CAST_OP(int32_t, cast_i32_f64, double);
CAST_OP(int32_t, cast_i32_f32, float);
CAST_OP(int32_t, cast_i32_i64, int64_t);
CAST_OP(int32_t, cast_i32_u32, u_int32_t);
CAST_OP(int32_t, cast_i32_u8, u_int8_t);

// I64
CAST_OP(int64_t, cast_i64_f64, double);
CAST_OP(int64_t, cast_i64_f32, float);
CAST_OP(int64_t, cast_i64_i32, int32_t);
CAST_OP(int64_t, cast_i64_u32, u_int32_t);
CAST_OP(int64_t, cast_i64_u8, u_int8_t);

// U32
CAST_OP(u_int32_t, cast_u32_f64, double);
CAST_OP(u_int32_t, cast_u32_f32, float);
CAST_OP(u_int32_t, cast_u32_i32, int32_t);
CAST_OP(u_int32_t, cast_u32_i64, int64_t);
CAST_OP(u_int32_t, cast_u32_u8, u_int8_t);

// U8
CAST_OP(u_int8_t, cast_u8_f64, double);
CAST_OP(u_int8_t, cast_u8_f32, float);
CAST_OP(u_int8_t, cast_u8_i32, int32_t);
CAST_OP(u_int8_t, cast_u8_i64, int64_t);
CAST_OP(u_int8_t, cast_u8_u32, u_int32_t);