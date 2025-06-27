#include "fields/alt_bn128-fp2.cuh"
#include "curves/jacobian_t.cuh"

namespace alt_bn128 {
    typedef jacobian_t<fp_t> g1_t;
    typedef jacobian_t<fp2_t> g2_t;
}

template <class field_t>
__device__ inline bool get_bit(const field_t& scalar, size_t i)
{
    return reinterpret_cast<const uint32_t*>(&scalar)[i / 32] & (1u << (i % 32));
}

template <class field_t, class jacobian_t>
__device__ inline jacobian_t scalar_mult(const field_t& scalar, const jacobian_t& point)
{
    jacobian_t result; result.inf();
    jacobian_t addend = point;
    field_t mut_scalar = scalar; mut_scalar.from();

    for (size_t i = 0; i < field_t::nbits; ++i) {
        jacobian_t update = result; update.add(addend);
        vec_select(&result, &update, &result, sizeof(jacobian_t), get_bit(mut_scalar, i));
        addend.dbl();
    }

    return result;
}