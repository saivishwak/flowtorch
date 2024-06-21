#include <math.h>

// TODO : Support column major format
#define UNARY_OP(TYPENAME, FN_NAME, FUNC)                                                    \
  extern "C" __global__ void FN_NAME(const size_t numel, const TYPENAME *lhs, TYPENAME *out) \
  {                                                                                          \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;                  \
         i += blockDim.x * gridDim.x)                                                        \
    {                                                                                        \
      TYPENAME x = lhs[i];                                                                   \
      out[i] = FUNC;                                                                         \
    }                                                                                        \
  }

// Neg
UNARY_OP(float, uneg_f32, -1 * x);
UNARY_OP(double, uneg_f64, -1 * x);
UNARY_OP(int, uneg_i32, -1 * x);
UNARY_OP(long long, uneg_i64, -1 * x);

// Sqr
UNARY_OP(float, usqr_f32, x *x);
UNARY_OP(double, usqr_f64, x *x);
UNARY_OP(int, usqr_i32, x *x);
UNARY_OP(long long, usqr_i64, x *x);

// Sqrt
UNARY_OP(float, usqrt_f32, sqrt(x));
UNARY_OP(double, usqrt_f64, sqrt(x));

// Ceil
UNARY_OP(float, uceil_f32, ceil(x));
UNARY_OP(double, uceil_f64, ceil(x));

// Floor
UNARY_OP(float, ufloor_f32, floor(x));
UNARY_OP(double, ufloor_f64, floor(x));