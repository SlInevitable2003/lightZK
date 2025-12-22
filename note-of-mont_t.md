# 1. 有限域运算: Montgomery乘法

```cpp
friend inline mont_t operator*(const mont_t& a, const mont_t& b)
{
    if (N%32 == 0) return wide_t{a, b};
    else {
        mont_t even, odd;

        #pragma unroll
        for (size_t i = 0; i < n; i += 2) {
            mad_n_redc(&even[0], &odd[0], &a[0], b[i], i==0);
            mad_n_redc(&odd[0], &even[0], &a[0], b[i+1]);
        }

        // merge |even| and |odd|
        cadd_n(&even[0], &odd[1], n-1);
        asm("addc.u32 %0, %0, 0;" : "+r"(even[n-1]));

        even.final_sub(0, &odd[0]);

        return even;
    }
}
```

## 1.1. 子模块1: 交错的标量乘法累加与模运算

```cpp
static inline void mad_n_redc(uint32_t *even, uint32_t* odd,
                              const uint32_t *a, uint32_t bi, 
                              bool first=false)
{
    if (first) {
        mul_n(odd, a+1, bi);
        mul_n(even, a,  bi);
    } else {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
        madc_n_rshift(odd, a+1, bi);
        cmad_n(even, a, bi);
        asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
    }

    uint32_t mi = even[0] * M0;

    cmad_n(odd, MOD+1, mi);
    cmad_n(even, MOD,  mi);
    asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
}
```
其中:
- `mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n_=n)`实现保持大整数$a[0:n-1]$的偶数字而置奇数字为$0$后, 计算其与$bi$的乘积并**赋**给$acc$.
- `cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n_=n)`实现保持大整数$a[0:n-1]$的偶数字而置奇数字为$0$后, 计算其与$bi$的乘积并**累加**到$acc$.

## 1.2. 子模块2: 带右移的标量乘法累加

```cpp
static inline void madc_n_rshift(uint32_t* odd, const uint32_t *a, uint32_t bi)
{
    for (size_t j = 0; j < n-2; j += 2)
        asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
            : "=r"(odd[j]), "=r"(odd[j+1])
            : "r"(a[j]), "r"(bi), "r"(odd[j+2]), "r"(odd[j+3]));
    asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
        : "=r"(odd[n-2]), "=r"(odd[n-1])
        : "r"(a[n-2]), "r"(bi));
}
```
- 循环体内的语句实现
$$
    (C,odd[j+1],odd[j])\gets(odd[j+3],odd[j+2])+a[j]\times bi+C
$$


# 2. 运行实例

1. 输入配置:

- $a = \{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]\}$
- $b = \{b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]\}$

2. 计算过程

- 创建变量$even, odd$
- $odd\gets\{a[1], 0, a[3], 0, a[5], 0, a[7], 0\}\times b[0]$
- $even\gets\{a[0], 0, a[2], 0, a[4], 0, a[6], 0\}\times b[0]$
- $m_0\gets(even[0]\times M_0)\text{ mod }D$
- $odd\gets odd+\{p[1], 0, p[3], 0, p[5], 0, p[7], 0\}\times m_0$
- $even\gets even+\{p[0], 0, p[2], 0, p[4], 0, p[6], 0\}\times m_0$
- 保证$odd$不发生溢出并将$even$的最高进位保存在$odd[7]$, 此时成立
$$
    even+odd\times D=a\times b[0]+p\times m_0=A'
$$
- $odd[0]\gets odd[0]+even[1]$
- $even\gets even/D^2+\{a[1], 0, a[3], 0, a[5], 0, a[7], 0\}\times b[1]$
- $odd\gets odd+\{a[0], 0, a[2], 0, a[4], 0, a[6], 0\}\times b[1]$
- 将$odd$的最高进位保存在$even[7]$
- $m_1\gets(even[0]\times M_0)\text{ mod }D$
- $even\gets even+\{p[1], 0, p[3], 0, p[5], 0, p[7], 0\}\times m_1$
- $odd\gets odd+\{p[0], 0, p[2], 0, p[4], 0, p[6], 0\}\times m_1$
- 保证$even$不发生溢出并将$odd$的最高进位保存在$even[7]$, 此时成立
$$
    odd+even\times D=A'/D+a\times b[1]+p\times m_1=A'
$$