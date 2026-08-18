#pragma once
#include "msm/msm_common.cuh"

template <typename FieldT, typename AffT, typename ProjT>
__global__ void fixmsm_kernel(const AffT *base, const FieldT *scalars, ProjT *out, size_t scale)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= scale) return;

    FieldT scalar = scalars[i];
    scalar.from();

    const AffT base_point = base[0];

    ProjT res;
    res.inf();
    for (int k = FieldT::nbits - 1; k >= 0; k--) {
        res.dbl();
        if (get_bit(scalar, k)) res.add(base_point);
    }
    out[i] = res;
}
