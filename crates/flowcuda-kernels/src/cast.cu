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
CAST_OP(double, cast_f64_f32, float);