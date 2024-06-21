template <typename T>
__device__ void fill_with(T *buf, T value, const size_t numel)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
       i += blockDim.x * gridDim.x)
  {
    buf[i] = value;
  }
}
extern "C" __global__ void fill_f32(float *buf, float value,
                                    const size_t numel)
{
  fill_with(buf, value, numel);
}
extern "C" __global__ void fill_f64(double *buf, double value,
                                    const size_t numel)
{
  fill_with(buf, value, numel);
}
extern "C" __global__ void fill_i64(int64_t *buf, int64_t value,
                                    const size_t numel)
{
  fill_with(buf, value, numel);
}

extern "C" __global__ void fill_i32(int32_t *buf, int32_t value,
                                    const size_t numel)
{
  fill_with(buf, value, numel);
}

extern "C" __global__ void fill_u32(u_int32_t *buf, u_int32_t value,
                                    const size_t numel)
{
  fill_with(buf, value, numel);
}
extern "C" __global__ void fill_u8(u_int8_t *buf, u_int8_t value,
                                   const size_t numel)
{
  fill_with(buf, value, numel);
}
