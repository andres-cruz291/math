
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


constexpr std::size_t m = 3;
constexpr std::size_t n = 4;

using namespace boost::math::differentiation;

/**
 * CUDA Kernels Device code
 *
 */
template <typename T, typename T1>
BOOST_MATH_CUDA_ENABLED void fill_output_sv(T variable, T1 *out, uint i, uint offset, uint ft){
    for (uint j = 0; j < ft; ++j)
        out[i * offset + j] = variable.derivative(j);
}

template <typename T, typename T1>
BOOST_MATH_CUDA_ENABLED void fill_output_dv(T variable, T1 *out, uint i, uint offset, uint ft, uint st){
    for (uint j = 0; j < ft; ++j)
        for (uint k = 0; k < st; ++k)
            out[i * offset + (j * (n + 1)) + k] = variable.derivative(j, k);
}

 template <typename T>
__global__ void construct_empty_single_variable(T *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const autodiff_fvar<T, m> empty = autodiff_fvar<T, m>();
        const uint offset = m + 1;
        fill_output_sv(empty, out, i, offset, offset);
    }
}

template <typename T>
__global__ void construct_empty_second_independent_variable(T *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const autodiff_fvar<T, m, n> empty = autodiff_fvar<T, m, n>();
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(empty, out, i, offset, m + 1, n + 1);
    }
}

template <typename T>
__global__ void construct_single_variable(T *out, int numElements, T value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const auto x = make_fvar<T, m>(value);
        const uint offset = m + 1;
        fill_output_sv(x, out, i, offset, offset);
    }
}

template <typename T>
__global__ void construct_second_independent_variable(T *out, int numElements, T value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const auto x = make_fvar<T, m, n>(value);
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(x, out, i, offset, m + 1, n + 1);
    }
}

template <typename T>
BOOST_MATH_CUDA_ENABLED T uncast_return(const T& x) {
  return x == 0 ? 0 : 1;
}

template <typename T>
__global__ void implicit_constructors(T *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        const autodiff_fvar<T, m> x = 3;
        const autodiff_fvar<T, m> one = uncast_return(x);
        const autodiff_fvar<T, m> two_and_a_half = 2.5;
        const uint offset = 3;
        out[i * offset] = static_cast<T>(x);
        out[i * offset + 1] = static_cast<T>(one);
        out[i * offset + 2] = static_cast<T>(two_and_a_half);
    }
}

template <typename T>
__global__ void assigment_to_empty_variable(T *out, int numElements, float cx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        autodiff_fvar<T, m, n> empty;
        auto x = make_fvar<T, m>(cx);
        empty = static_cast<decltype(empty)>(x);
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(empty, out, i, offset, m + 1, n + 1);
    }
}

template <typename T>
__global__ void default_assigment_operator(T *out, int numElements, float cx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        autodiff_fvar<T, m, n> empty;
        auto x = make_fvar<T, m, n>(cx);
        empty = x;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(empty, out, i, offset, m + 1, n + 1);
    }
}

template <typename T>
__global__ void assigment_constant_to_empty_variable(T *out, int numElements, float cx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        autodiff_fvar<T, m, n> empty;
        empty = cx;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(empty, out, i, offset, m + 1, n + 1);
    }
}

template <typename T>
__global__ void addition_assignment_single_variable(T *out, int numElements, float cx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        auto sum = autodiff_fvar<T, m, n>(); // zero-initialized
        // Single variable
        const auto x = make_fvar<T, m>(cx);
        sum += x;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(sum, out, i, offset, m + 1, n + 1);
    }
}

template <typename T>
__global__ void addition_assignment_arithmetic_constant(T *out, int numElements, float cx)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements){
        auto sum = autodiff_fvar<T, m, n>(); // zero-initialized
        sum = 0;
        sum += cx;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(sum, out, i, offset, m + 1, n + 1);
    }
}

/**
 * Host main routines
 */
template <typename T>
bool construct_empty_single_variable(uint numElements)
{
    uint outputSize = numElements * (m + 1);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "construct_empty_single_variable", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "construct_empty_single_variable"))
        return false;
    
    watch w;

    // Execute the kernel
    construct_empty_single_variable<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "construct_empty_single_variable"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool construct_empty_second_independent_variable(uint numElements)
{
    uint outputSize = numElements * (m + 1) * (n + 1);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "construct_empty_second_independent_variable", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "construct_empty_second_independent_variable"))
        return false;
    
    watch w;

    // Execute the kernel
    construct_empty_second_independent_variable<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "construct_empty_second_independent_variable"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool construct_single_variable(uint numElements)
{
    uint outputSize = numElements * (m + 1);
    const T cx = T(10);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "construct_single_variable", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "construct_single_variable"))
        return false;
    
    watch w;

    // Execute the kernel
    construct_single_variable<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "construct_single_variable"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += m + 1){
        referenceData[0][i] = cx;
        referenceData[0][i + 1] = T(1);
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool construct_second_independent_variable(uint numElements)
{
    uint outputSize = numElements * (m + 1) * (n + 1);
    const T cx = T(10);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "construct_second_independent_variable", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "construct_second_independent_variable"))
        return false;
    
    watch w;

    // Execute the kernel
    construct_second_independent_variable<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "construct_second_independent_variable"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += (m + 1) * (n + 1)){
        referenceData[0][i] = cx;
        referenceData[0][i + 1] = T(1);
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool implicit_constructors(uint numElements)
{
    uint outputSize = numElements * 3;

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "implicit_constructors", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "implicit_constructors"))
        return false;
    
    watch w;

    // Execute the kernel
    implicit_constructors<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "implicit_constructors"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize);
    for (uint i = 0; i < numElements; ++i){
        referenceData[0][i * 3] = static_cast<T>(3.0);
        referenceData[0][i * 3 + 1] = static_cast<T>(1.0);
        referenceData[0][i * 3 + 2] = static_cast<T>(2.5);
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool assigment_to_empty_variable(uint numElements)
{
    uint outputSize = numElements * (m + 1) * (n + 1);
    const T cx = T(10);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "assigment_to_empty_variable", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "assigment_to_empty_variable"))
        return false;
    
    watch w;

    // Execute the kernel
    assigment_to_empty_variable<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "assigment_to_empty_variable"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += (m + 1) * (n + 1)){
        referenceData[0][i] = cx;
        referenceData[0][i + (n + 1)] = T(1);
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool default_assigment_operator(uint numElements)
{
    uint outputSize = numElements * (m + 1) * (n + 1);
    const T cx = T(10);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "default_assigment_operator", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "default_assigment_operator"))
        return false;
    
    watch w;

    // Execute the kernel
    default_assigment_operator<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "default_assigment_operator"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += (m + 1) * (n + 1)){
        referenceData[0][i] = cx;
        referenceData[0][i + 1] = T(1);
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool assigment_constant_to_empty_variable(uint numElements)
{
    uint outputSize = numElements * (m + 1) * (n + 1);
    const T cx = T(10);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "assigment_constant_to_empty_variable", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "assigment_constant_to_empty_variable"))
        return false;
    
    watch w;

    // Execute the kernel
    assigment_constant_to_empty_variable<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "assigment_constant_to_empty_variable"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += (m + 1) * (n + 1)){
        referenceData[0][i] = cx;
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool addition_assignment_single_variable(uint numElements)
{
    uint outputSize = numElements * (m + 1) * (n + 1);
    const T cx = T(10);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "addition_assignment_single_variable", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "addition_assignment_single_variable"))
        return false;
    
    watch w;

    // Execute the kernel
    addition_assignment_single_variable<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "addition_assignment_single_variable"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += (m + 1) * (n + 1)){
        referenceData[0][i] = cx;
        referenceData[0][i + (n + 1)] = T(1);
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool addition_assignment_arithmetic_constant(uint numElements)
{
    uint outputSize = numElements * (m + 1) * (n + 1);
    const T cx = T(11);

    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( "addition_assignment_arithmetic_constant", numElements, {}, { outputSize });

    if (!verifyCudaStatus("prepare", "addition_assignment_arithmetic_constant"))
        return false;
    
    watch w;

    // Execute the kernel
    addition_assignment_arithmetic_constant<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;

    if (!verifyCudaStatus("launch", "addition_assignment_arithmetic_constant"))
        return false;
    
    // Verify that the result vector is correct
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    std::vector<std::vector<T>> referenceData(1);
    referenceData[0].resize(outputSize, T(0));
    for (uint i = 0; i < outputSize; i += (m + 1) * (n + 1)){
        referenceData[0][i] = cx;
    }
    
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;

    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename float_type>
int main_tests_1(uint numElements){
    if (!construct_empty_single_variable<float_type>(numElements))
        return EXIT_FAILURE;
    if (!construct_empty_second_independent_variable<float_type>(numElements))
        return EXIT_FAILURE;
    if (!construct_single_variable<float_type>(numElements))
        return EXIT_FAILURE;
    if (!construct_second_independent_variable<float_type>(numElements))
        return EXIT_FAILURE;
    if (!implicit_constructors<float_type>(numElements))
        return EXIT_FAILURE;
    if (!assigment_to_empty_variable<float_type>(numElements))
        return EXIT_FAILURE;
    if (!default_assigment_operator<float_type>(numElements))
        return EXIT_FAILURE;
    if (!assigment_constant_to_empty_variable<float_type>(numElements))
        return EXIT_FAILURE;
    if (!addition_assignment_single_variable<float_type>(numElements))
        return EXIT_FAILURE;
    if (!addition_assignment_arithmetic_constant<float_type>(numElements))
        return EXIT_FAILURE;
        
    return 0;
}