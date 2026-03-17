
//  Copyright Andres Cruz 2026.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/math/special_functions.hpp>
#include "stopwatch.hpp"
#include <boost/math/differentiation/autodiff.hpp>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "test_autodiff_cuda.hpp"

using namespace boost::math::differentiation;

/**
 * CUDA Kernels Device code
 *
 */
 template <typename T>
__global__ void cast_double(T *out, int numElements, T ca, T j)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const auto x = autodiff_fvar<T, m>(ca);
        const uint offset = 2;
        out[i * offset] = static_cast<T>(x);
        out[i * offset + 1] = static_cast<T>(j * x);
    }
}

template <typename T>
__global__ void int_double_casting(T *out, int numElements, T ca)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const auto x = autodiff_fvar<T, m>(ca);
        const uint offset = 3;
        const auto x0 = make_fvar<T, 0>(ca);
        out[i * offset] = static_cast<T>(x0);
        const auto x1 = make_fvar<T, 1>(ca);
        out[i * offset + 1] = static_cast<T>(x1);
        const auto x2 = make_fvar<T, 2>(ca);
        out[i * offset + 2] = static_cast<T>(x2);
    }
}

template <typename T>
__global__ void scalar_addition(T *out, int numElements, T ca, T cb)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const auto x = autodiff_fvar<T, m>(ca);
        const uint offset = 3;
        const auto sum0 = autodiff_fvar<T, 0>(ca) + autodiff_fvar<T, 0>(cb);
        out[i * offset] = static_cast<T>(sum0);
        const auto sum1 = autodiff_fvar<T, 0>(ca) + cb;
        out[i * offset + 1] = static_cast<T>(sum1);
        const auto sum2 = ca + autodiff_fvar<T, 0>(cb);
        out[i * offset + 2] = static_cast<T>(sum2);
    }
}

template <typename T, int n>
__global__ void power8(T *out1, T *out2, int numElements, T ca)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const uint offset = n + 1;
        
        // Test operator*=()
        auto x = make_fvar<T, n>(ca);
        x *= x;
        x *= x;
        x *= x;
        fill_output_sv(x, out1, i, offset, offset);
        
        // Test operator*()
        x = make_fvar<T, n>(ca);
        x = x * x * x * x * x * x * x * x;
        fill_output_sv(x, out2, i, offset, offset);
    }
}

template <typename T, int m, int n>
__global__ void dim1_multiplication(T *out1, T *out2, int numElements, T cy)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const uint offset = n + 1;
        
        auto y0 = make_fvar<T, m>(cy);
        auto y = make_fvar<T, n>(cy);
        y *= y0;
        fill_output_sv(y, out1, i, offset, offset);
        
        y = y * cy;
        fill_output_sv(y, out2, i, offset, offset);
    }
}

/**
 * Host main routines
 */
template <typename T>
bool cast_double(uint numElements)
{
    uint outputSize = numElements * 2;
    const T ca(13);
    const T j(12);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "cast_double", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "cast_double"))
        return false;
    
    watch w;

    // Execute the kernel
    cast_double<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, ca, j);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "cast_double"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    bool error = false;
    for (uint i = 0; i < outputSize; i += 2){
        if (i % 2 == 0){
            if (!(j < h_out[0][i])){
                std::cerr << "Result verification failed at element " << i << "! Not lt Value: "<< h_out[0][i] << std::endl;
                error = true;
            }
        }else if(h_out[0][i] != i * ca){
            std::cerr << "Result verification failed at element " << i << "! Value: "<< h_out[0][i] << std::endl;
            error = true;
        }
    }
    
    if (error)
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool int_double_casting(uint numElements)
{
    uint outputSize = numElements * 3;
    const T cx = 3;

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "int_double_casting", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "int_double_casting"))
        return false;
    
    watch w;

    // Execute the kernel
    int_double_casting<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "int_double_casting"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, cx);
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool scalar_addition(uint numElements)
{
    uint outputSize = numElements * 3;
    const T ca = 3;
    const T cb = 4;

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "scalar_addition", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "scalar_addition"))
        return false;
    
    watch w;

    // Execute the kernel
    scalar_addition<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, ca, cb);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "scalar_addition"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, (ca + cb));
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool power8(uint numElements)
{
    constexpr std::size_t n = 8u;
    uint outputSize = numElements * (n + 1);
    const T cx = T(3);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "power8", numElements, {}, { outputSize, outputSize });

    if (!verifyCudaStatus("prepare", "power8"))
        return false;
    
    watch w;

    // Execute the kernel
    power8<T, n><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "power8"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(2);
    referenceData[0].resize(outputSize);
    referenceData[1].resize(outputSize);
    const T power_factorial = boost::math::factorial<T>(n);
    for (uint i = 0; i < outputSize; ++i){
        uint j = i % (n + 1);
        referenceData[0][i] = power_factorial / 
                              boost::math::factorial<T>(static_cast<unsigned>(n - j)) * 
                              pow(cx, n - j);
        referenceData[1][i] = referenceData[0][i];
    }
    
    if (!verifyCloseOutputs(h_out, referenceData, std::numeric_limits<T>::epsilon()))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool dim1_multiplication(uint numElements)
{
    constexpr std::size_t m = 2;
    constexpr std::size_t n = 3;
    
    uint outputSize = numElements * (n + 1);
    const T cy = 4;

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "dim1_multiplication", numElements, {}, { outputSize, outputSize });

    if (!verifyCudaStatus("prepare", "dim1_multiplication"))
        return false;
    
    watch w;

    // Execute the kernel
    dim1_multiplication<T, m, n><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], numElements, cy);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "dim1_multiplication"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(2);
    referenceData[0].resize(outputSize, T(0));
    referenceData[1].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += (n + 1)){
        referenceData[0][i] = cy * cy;
        referenceData[0][i + 1] = 2 * cy;
        referenceData[0][i + 2] = T(2);

        referenceData[1][i] = cy * cy * cy;
        referenceData[1][i + 1] = 2 * cy * cy;
        referenceData[1][i + 2] = 2 * cy;
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename float_type>
bool main_tests_2(uint numElements){
    if (!cast_double<float_type>(numElements))
        return false;
    if (!int_double_casting<float_type>(numElements))
        return false;
    if (!scalar_addition<float_type>(numElements))
        return false;
    if (!power8<float_type>(numElements))
        return false;
    if (!dim1_multiplication<float_type>(numElements))
        return false;

    return true;
}