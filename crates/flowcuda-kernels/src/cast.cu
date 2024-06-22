#include <math.h>

// TODO : Support column major format
template <typename T, typename S>
__device__ void cast(const size_t numel, const T *data, S *out)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x)
    {
        out[i] = data[i];
    }
}

#define CAST_OP(TYPENAME, FN_NAME, TO_TYPENAME)                                                    \
    extern "C" __global__ void FN_NAME(const size_t numel, const TYPENAME *data, TO_TYPENAME *out) \
    {                                                                                              \
        cast<TYPENAME, TO_TYPENAME>(numel, data, out);                                             \
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