#define BINARY_OP(TYPENAME, FN_NAME, FUNC)                                    \
  extern "C" __global__ void FN_NAME(const size_t numel, const TYPENAME *lhs, \
                                     const TYPENAME *rhs, TYPENAME *out)      \
  {                                                                           \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;   \
         i += blockDim.x * gridDim.x)                                         \
    {                                                                         \
      TYPENAME x = lhs[i];                                                    \
      TYPENAME y = rhs[i];                                                    \
      out[i] = FUNC;                                                          \
    }                                                                         \
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
