# lightZK

## Requirements

1. GCC
2. CMake
3. GNU GMP
4. Boost
5. CUDA Toolkit
6. OpenSSL
7. OpenMP 

## Build

1. 在CMakeLists.txt中修改`set(CMAKE_CUDA_ARCHITECTURES 120)`并设置为目标架构（例如NVIDIA A100 GPU为`80`）。
2. `cd build && cmake .. && make`

## Run

### MSM标准测试

`test/msm-test.cu`包含MSM标准测试，覆盖数据生成、计算和对比验证。计算规模硬编码于`MSMTest<ppT> msm_test(1 << 22, pregen_option == "-fast");`中。

运行命令为
```
cd build && ./msm-test --regen % 首次生成数据并保存在本地
% 或
cd build && ./msm-test --fast % 已经生成了数据，因此直接使用本地的数据计算
```
