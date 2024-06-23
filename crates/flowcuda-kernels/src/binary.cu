#include "utils.cuh"

// TODO : Refactor
#define BINARY_OP(TYPENAME, FN_NAME, FUNC)                                                                                                                                                                 \
  extern "C" __global__ void FN_NAME(const size_t numel, const TYPENAME *lhs,                                                                                                                              \
                                     const TYPENAME *rhs, TYPENAME *out, bool lhs_is_contiguous, size_t *lhs_layout, size_t lhs_num_dims, bool rhs_is_contiguous, size_t *rhs_layout, size_t rhs_num_dims) \
  {                                                                                                                                                                                                        \
    if (rhs_is_contiguous && lhs_is_contiguous)                                                                                                                                                            \
    {                                                                                                                                                                                                      \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)                                                                                                 \
      {                                                                                                                                                                                                    \
        TYPENAME x = lhs[i];                                                                                                                                                                               \
        TYPENAME y = rhs[i];                                                                                                                                                                               \
        out[i] = FUNC;                                                                                                                                                                                     \
      }                                                                                                                                                                                                    \
    }                                                                                                                                                                                                      \
    else if (lhs_is_contiguous)                                                                                                                                                                            \
    {                                                                                                                                                                                                      \
      size_t *rhs_dims = rhs_layout;                                                                                                                                                                       \
      size_t *rhs_strides = rhs_dims + rhs_num_dims;                                                                                                                                                       \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)                                                                                                 \
      {                                                                                                                                                                                                    \
        unsigned strided_i = get_strided_index(i, rhs_num_dims, rhs_dims, rhs_strides);                                                                                                                    \
        TYPENAME x = lhs[i];                                                                                                                                                                               \
        TYPENAME y = rhs[strided_i];                                                                                                                                                                       \
        out[i] = FUNC;                                                                                                                                                                                     \
      }                                                                                                                                                                                                    \
    }                                                                                                                                                                                                      \
    else if (rhs_is_contiguous)                                                                                                                                                                            \
    {                                                                                                                                                                                                      \
      size_t *lhs_dims = lhs_layout;                                                                                                                                                                       \
      size_t *lhs_strides = lhs_dims + lhs_num_dims;                                                                                                                                                       \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)                                                                                                 \
      {                                                                                                                                                                                                    \
        unsigned strided_i = get_strided_index(i, lhs_num_dims, lhs_dims, lhs_strides);                                                                                                                    \
        TYPENAME x = lhs[strided_i];                                                                                                                                                                       \
        TYPENAME y = rhs[i];                                                                                                                                                                               \
        out[i] = FUNC;                                                                                                                                                                                     \
      }                                                                                                                                                                                                    \
    }                                                                                                                                                                                                      \
    else                                                                                                                                                                                                   \
    {                                                                                                                                                                                                      \
      size_t *lhs_dims = lhs_layout;                                                                                                                                                                       \
      size_t *lhs_stride = lhs_dims + lhs_num_dims;                                                                                                                                                        \
      size_t *rhs_dims = rhs_layout;                                                                                                                                                                       \
      size_t *rhs_stride = rhs_dims + rhs_num_dims;                                                                                                                                                        \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)                                                                                                 \
      {                                                                                                                                                                                                    \
        unsigned lhs_strided_i = get_strided_index(i, lhs_num_dims, lhs_dims, lhs_stride);                                                                                                                 \
        unsigned rhs_strided_i = get_strided_index(i, rhs_num_dims, rhs_dims, rhs_stride);                                                                                                                 \
        TYPENAME x = lhs[lhs_strided_i];                                                                                                                                                                   \
        TYPENAME y = rhs[rhs_strided_i];                                                                                                                                                                   \
        out[i] = FUNC;                                                                                                                                                                                     \
      }                                                                                                                                                                                                    \
    }                                                                                                                                                                                                      \
  }

// ADD
BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_f64, x + y);
BINARY_OP(int, badd_i32, x + y)
BINARY_OP(long long, badd_i64, x + y);

// DUV
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y);
BINARY_OP(int, bdiv_i32, x / y)
BINARY_OP(long long, bdiv_i64, x / y);

// MULL
BINARY_OP(float, bmul_f32, x *y)
BINARY_OP(double, bmul_f64, x *y);
BINARY_OP(int, bmul_i32, x *y)
BINARY_OP(long long, bmul_i64, x *y);

// SUB
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
BINARY_OP(int, bsub_i32, x - y)
BINARY_OP(long long, bsub_i64, x - y);

// MAX
BINARY_OP(float, bmax_f32, x > y ? x : y)
BINARY_OP(double, bmax_f64, x > y ? x : y);
BINARY_OP(int, bmax_i32, x > y ? x : y)
BINARY_OP(long long, bmax_i64, x > y ? x : y);

// MIN
BINARY_OP(float, bmin_f32, x < y ? x : y)
BINARY_OP(double, bmin_f64, x < y ? x : y);
BINARY_OP(int, bmin_i32, x < y ? x : y)
BINARY_OP(long long, bmin_i64, x < y ? x : y);
