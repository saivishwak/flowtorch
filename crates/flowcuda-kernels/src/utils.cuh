__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides)
{
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++)
    {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}