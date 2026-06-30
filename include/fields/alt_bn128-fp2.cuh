// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "alt_bn128.cuh"

# define inline __device__ __forceinline__
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

# ifndef WARP_SZ
#  define WARP_SZ 32
# endif

namespace alt_bn128 {

// class fp2_t : public fp_mont {
// private:
//     static inline uint32_t laneid()
//     {   return threadIdx.x % WARP_SZ;   }

// public:
//     static const uint32_t degree = 2;

//     class mem_t { friend fp2_t;
//         fp_mont x[2];

//     public:
//         inline operator fp2_t() const           { return x[threadIdx.x&1]; }
//         inline void zero()                      { x[threadIdx.x&1].zero(); }
//         inline void to()                        { x[threadIdx.x&1].to();   }
//         inline void from()                      { x[threadIdx.x&1].from(); }
//         inline mem_t& operator=(const fp2_t& a)
//         {   x[threadIdx.x&1] = a; return *this;   }
//     };

//     inline fp2_t()                              {}
//     inline fp2_t(const fp_mont& a) : fp_mont(a) {}
//     inline fp2_t(const mem_t* p)                { *this = p->x[threadIdx.x&1]; }
//     inline void store(mem_t* p) const           { p->x[threadIdx.x&1] = *this; }

//     friend inline fp2_t operator*(const fp2_t& a, const fp2_t& b)
//     {
//         auto id = laneid();
//         auto mask = __activemask();
//         auto t0 = b.shfl(id&~1, mask);
//         auto t1 = a.shfl(id^1, mask);
//         auto t2 = b.shfl(id|1, mask);
//         t1.cneg((id&1) == 0);

//         return a * t0 + t1 * t2;  // a*t0 + t1*t2;
//     }
//     inline fp2_t& operator*=(const fp2_t& a)
//     {   return *this = *this * a;   }

//     inline fp2_t& sqr()
//     {
//         auto id = laneid();
//         fp_mont t0 = shfl(id^1, __activemask());
//         fp_mont t1 = *this;

//         if ((id&1) == 0) {
//             t1 = (fp_mont)*this + t0;
//             t0 = (fp_mont)*this - t0;
//         }
//         t0 *= t1;
//         t1 = t0 << 1;

//         return *this = fp_mont::csel(t1, t0, id&1);
//     }
//     inline fp2_t& operator^=(int p)
//     {   if (p != 2) asm("trap;"); return sqr();     }
//     friend inline fp2_t operator^(fp2_t a, int p)
//     {   if (p != 2) asm("trap;"); return a.sqr();   }

//     friend inline fp2_t operator+(const fp2_t& a, const fp2_t& b)
//     {   return (fp_mont)a + (fp_mont)b;   }
//     inline fp2_t& operator+=(const fp2_t& b)
//     {   return *this = *this + b;   }

//     friend inline fp2_t operator-(const fp2_t& a, const fp2_t& b)
//     {   return (fp_mont)a - (fp_mont)b;   }
//     inline fp2_t& operator-=(const fp2_t& b)
//     {   return *this = *this - b;   }

//     friend inline fp2_t operator<<(const fp2_t& a, unsigned l)
//     {   return (fp_mont)a << l;   }
//     inline fp2_t& operator<<=(unsigned l)
//     {   return *this = *this << l;   }

//     inline fp2_t& cneg(bool flag)
//     {   fp_mont::cneg(flag); return *this;  }
//     friend inline fp2_t cneg(fp2_t a, bool flag)
//     {   return a.cneg(flag);   }

//     friend inline fp2_t czero(const fp2_t& a, int set_z)
//     {   return czero((fp_mont)a, set_z);   }

//     inline bool is_zero() const
//     {
//         auto ret = __ballot_sync(__activemask(), fp_mont::is_zero());
//         return ((ret >> (laneid()&~1)) & 3) == 3;
//     }

//     inline bool is_zero(const fp2_t& a) const
//     {
//         auto ret = __ballot_sync(__activemask(), fp_mont::is_zero(a));
//         return ((ret >> (laneid()&~1)) & 3) == 3;
//     }

//     static inline fp2_t one(int or_zero = 0)
//     {   return fp_mont::one((laneid()&1) | or_zero);   }

//     inline bool is_one() const
//     {
//         auto id = laneid();
//         auto even = ~(0 - (id&1));
//         uint32_t is_zero = ((fp_mont)*this)[0] ^ (fp_mont::one()[0] & even);

//         for (size_t i = 1; i < n; i++)
//             is_zero |= ((fp_mont)*this)[i] ^ (fp_mont::one()[i] & even);

//         is_zero = __ballot_sync(__activemask(), is_zero == 0);
//         return ((is_zero >> (id&~1)) & 3) == 3;
//     }

//     inline fp2_t reciprocal() const
//     {
//         auto a = (fp_mont)*this^2;
//         auto b = shfl_xor(a);
//         a += b;
//         a = ct_inverse_mod_x(a);    // 1/(x[0]^2 + x[1]^2)
//         a *= (fp_mont)*this;
//         a.cneg(threadIdx.x&1);
//         return a;
//     }
//     friend inline fp2_t operator/(int one, const fp2_t& a)
//     {   if (one != 1) asm("trap;"); return a.reciprocal();   }
//     friend inline fp2_t operator/(const fp2_t& a, const fp2_t& b)
//     {   return a * b.reciprocal();   }
//     inline fp2_t& operator/=(const fp2_t& a)
//     {   return *this *= a.reciprocal();   }
// };

class fp2_t {
public:
    static const uint32_t degree = 2;

    fp_mont c0, c1;

    inline fp2_t() {}
    inline fp2_t(const fp_mont& a, const fp_mont& b) : c0(a), c1(b) {}

    class mem_t {
        friend fp2_t;
        fp_mont x[2];

    public:
        inline operator fp2_t() const { return fp2_t(x[0], x[1]); }
        inline void zero() { x[0].zero(); x[1].zero(); }
        inline void to() { x[0].to(); x[1].to(); }
        inline void from() { x[0].from(); x[1].from(); }

        inline mem_t& operator=(const fp2_t& a)
        {
            x[0] = a.c0;
            x[1] = a.c1;
            return *this;
        }
    };

    inline void zero() { c0.zero(); c1.zero(); }
    static inline fp2_t one(int or_zero = 0) { return fp2_t(fp_mont::one(or_zero), fp_mont::one(1)); }

    inline bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
    inline bool is_zero(const fp2_t& a) const { return a.c0.is_zero() && a.c1.is_zero(); }
    inline bool is_one()  const { return c0.is_one()  && c1.is_zero(); }

    friend inline fp2_t operator+(const fp2_t& a, const fp2_t& b)
    {   return fp2_t(a.c0 + b.c0, a.c1 + b.c1);   }
    inline fp2_t& operator+=(const fp2_t& b)
    {   c0 += b.c0; c1 += b.c1; return *this;   }

    friend inline fp2_t operator-(const fp2_t& a, const fp2_t& b)
    {   return fp2_t(a.c0 - b.c0, a.c1 - b.c1);   }
    inline fp2_t& operator-=(const fp2_t& b)
    {   c0 -= b.c0; c1 -= b.c1; return *this;   }

    friend inline fp2_t operator-(const fp2_t& a)
    {   return fp2_t(-a.c0, -a.c1);   }

    friend inline fp2_t operator*(const fp2_t& a, const fp2_t& b)
    {
        fp_mont m0 = a.c0 * b.c0;
        fp_mont m1 = a.c1 * b.c1;
        fp_mont m2 = (a.c0 + a.c1) * (b.c0 + b.c1);
        return fp2_t(m0 - m1, m2 - m0 - m1);
    }
    inline fp2_t& operator*=(const fp2_t& b)
    {   return *this = *this * b;   }

    inline fp2_t& sqr()
    {
        fp_mont t0 = c0 + c1;
        fp_mont t1 = c0 - c1;
        fp_mont t2 = c0 * c1;
        c0 = t0 * t1;
        c1 = t2 + t2;
        return *this;
    }
    friend inline fp2_t sqr(const fp2_t& a)
    {   fp2_t r = a; return r.sqr();   }
    inline fp2_t& operator^=(int p)
    {   if (p != 2) asm("trap;"); return sqr();   }
    friend inline fp2_t operator^(fp2_t a, int p)
    {   if (p != 2) asm("trap;"); return a.sqr();   }


    inline fp2_t& cneg(bool flag)
    {   c0.cneg(flag); c1.cneg(flag); return *this;   }
    friend inline fp2_t cneg(fp2_t a, bool flag)
    {   return a.cneg(flag);   }
    friend inline fp2_t czero(const fp2_t& a, int set_z)
    {   return fp2_t(czero(a.c0, set_z), czero(a.c1, set_z));   }

    friend inline fp2_t operator<<(const fp2_t& a, unsigned l)
    {   return fp2_t(a.c0 << l, a.c1 << l);   }
    inline fp2_t& operator<<=(unsigned l)
    {   c0 <<= l; c1 <<= l; return *this;   }

    inline fp2_t reciprocal() const
    {
        fp_mont asq = c0 * c0;
        fp_mont bsq = c1 * c1;
        fp_mont norm = 1 / (asq + bsq);
        return fp2_t(c0 * norm, -(c1 * norm));
    }
    friend inline fp2_t operator/(int one, const fp2_t& a)
    {   if (one != 1) asm("trap;"); return a.reciprocal();   }
    friend inline fp2_t operator/(const fp2_t& a, const fp2_t& b)
    {   return a * b.reciprocal();   }
    inline fp2_t& operator/=(const fp2_t& a)
    {   return *this = *this * a.reciprocal();   }
};


} // namespace alt_bn128

# undef inline
# undef asm

