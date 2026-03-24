//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_AUTODIFF_CUDA_HPP
#define BOOST_MATH_TEST_AUTODIFF_CUDA_HPP

#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MODULE test_autodiff_cuda
#endif

#ifndef BOOST_ALLOW_DEPRECATED_HEADERS
#define BOOST_ALLOW_DEPRECATED_HEADERS // artifact of sp_typeinfo.hpp inclusion from unit_test.hpp
#endif

#include <boost/math/tools/config.hpp>

#include <iostream>
#include <iomanip>
#include <vector>


constexpr std::size_t m = 3;
constexpr std::size_t n = 4;

template <typename T, typename T1>
BOOST_MATH_CUDA_ENABLED void fill_output_sv(T variable, T1 *out, uint i, uint offset, uint ft){
    for (uint j = 0; j < ft; ++j)
        out[i * offset + j] = variable.derivative(j);
}

template <typename T, typename T1>
BOOST_MATH_CUDA_ENABLED void fill_output_dv(T variable, T1 *out, uint i, uint offset, uint ft, uint st){
    for (uint j = 0; j < ft; ++j)
        for (uint k = 0; k < st; ++k)
            out[i * offset + (j * st) + k] = variable.derivative(j, k);
}

template <typename T, typename Op>
__global__ void apply_op(T *out, int numElements, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, int numElements, T cx, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, cx);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, int numElements, T cx, T cy, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, cx, cy);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, T *out1, int numElements, T cx, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, out1, cx);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, T *out1, int numElements, T cx, T cy, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, out1, cx, cy);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, T *out1, T *out2, int numElements, T cx, T cy, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, out1, out2, cx, cy);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, T *out1, T *out2, int numElements, T cx, T cy, T cz, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, out1, out2, cx, cy, cz);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, T *out1, T *out2, T *out3, int numElements, T cx, T cy, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, out1, out2, out3, cx, cy);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, T *out1, T *out2, T *out3, int numElements, T cx, T cy, T cz, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, out1, out2, out3, cx, cy, cz);
    }
}

template <typename T, typename Op>
__global__ void apply_op(T *out, T *out2, T *out3, T *out4, T* out5, int numElements, T cx, T cy, Op op)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        op(i, out, out2, out3, out4, out5, cx, cy);
    }
}

namespace diff = boost::math::differentiation;

template<typename T>
struct KernelLaunchConfig
{
    int numElements;
    std::vector<std::vector<T>> inputVars;
    std::vector<uint> h_outputVars;
    std::vector<uint> d_outputVars;
    int threadsPerBlock;
    int blocksPerGrid;
    std::vector<T*> d_inputs;
    std::vector<T*> d_outputs;
    std::vector<T*> h_outputs;
};

template<typename T>
KernelLaunchConfig<T> prepareKernelLaunch
(
    const std::string& testName
  , int numElements
  , std::vector<std::vector<T>> inputVars
  , std::vector<uint> outputVars
  , int threadsPerBlock = 32
)
{
    std::cout << "Test " << testName << std::endl;

    cudaError_t err = cudaSuccess;

    std::cout << "[Vector operation on " << numElements << " elements]" << std::endl;

    KernelLaunchConfig<T> config;

    config.numElements = numElements;
    config.inputVars = inputVars;
    config.h_outputVars = outputVars;
    config.d_outputVars = outputVars;
    config.threadsPerBlock = threadsPerBlock;
    config.blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "CUDA kernel launch with " << config.blocksPerGrid << " blocks of " << config.threadsPerBlock << " threads" << std::endl;

    // Allocate input buffers
    config.d_inputs.resize(inputVars.size());
    for (uint i = 0; i < inputVars.size(); ++i)
    {
        T h_inputValues[inputVars[i].size()];
        for (uint j = 0; j < inputVars[i].size(); ++j){
            h_inputValues[j] = inputVars[i][j];
        }
        err = cudaMalloc(&config.d_inputs[i], inputVars[i].size() * sizeof(T));
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to allocate input " << i << std::endl;
            exit(EXIT_FAILURE);
        }
        cudaMemcpy(config.d_inputs[i], h_inputValues, inputVars[i].size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Allocate output buffers
    config.d_outputs.resize(outputVars.size());
    for (int i = 0; i < outputVars.size(); ++i)
    {
        config.d_outputVars[i] *= numElements;
        err = cudaMalloc(&config.d_outputs[i], config.d_outputVars[i] * sizeof(T));
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to allocate output " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    config.h_outputs.resize(outputVars.size());
    for (uint i = 0; i < outputVars.size(); ++i){
        config.h_outputs[i] = new T[outputVars[i]];
    }
    return config;
}

bool verifyCudaStatus
(
    const std::string& testName
  , const std::string& step
){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to " << step << " " << testName << " kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        return false;
    }
    return true;
}

template<typename T>
std::vector<std::vector<T>> defineReferenceData(KernelLaunchConfig<T> cfg){
    std::vector<std::vector<T>> referenceData(cfg.h_outputVars.size());
    for (uint i = 0; i < cfg.h_outputVars.size(); ++i){
        referenceData[i].resize(cfg.d_outputVars[i]);
        for (uint j = 0; j < cfg.h_outputVars[i]; ++j)
            for (uint k = 0; k < cfg.numElements; ++k)
                referenceData[i][k * cfg.h_outputVars[i] + j] = cfg.h_outputs[i][j];
        
    }
    return referenceData;
}

template<typename T>
std::vector<std::vector<T>> returnOutputs(KernelLaunchConfig<T> cfg){
    std::vector<std::vector<T>> output(cfg.d_outputVars.size());
    for (uint i = 0; i < cfg.d_inputs.size(); ++i){
        cudaFree(cfg.d_inputs[i]);
    }
    for (uint i = 0; i < cfg.d_outputVars.size(); ++i){
        T h_output[cfg.d_outputVars[i]];
        output[i].resize(cfg.d_outputVars[i]);
        cudaMemcpy(h_output, cfg.d_outputs[i], cfg.d_outputVars[i] * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(cfg.d_outputs[i]);
        for (uint j = 0; j < cfg.d_outputVars[i]; ++j)
          output[i][j] = h_output[j];
    }
    return output;
}

template<typename T>
bool verifyEqualOutputs
( 
    std::vector<std::vector<T>> h_output
  , std::vector<std::vector<T>> referenceData
){
    bool valid = true;
    for(uint i = 0; i < h_output.size(); ++i){
        for(uint j = 0; j < h_output[i].size(); ++j){
            if (h_output[i][j] != referenceData[i][j]){
                std::cerr << "Result verification failed at element " << j << " of output " << i << "! Value: "<< h_output[i][j] << " Reference: " << referenceData[i][j] << std::endl;
                valid = false;
            }
        }
    }
    return valid;
}

template<typename T>
bool verifyCloseOutputs
( 
    std::vector<std::vector<T>> h_output
  , std::vector<std::vector<T>> referenceData
  , T tolerance_percent
){
    bool valid = true;
    for(uint i = 0; i < h_output.size(); ++i){
        for(uint j = 0; j < h_output[i].size(); ++j){
            if (h_output[i][j] == referenceData[i][j]){
                continue;
            }
            T diff = std::fabs(h_output[i][j] - referenceData[i][j]);
            T largest = std::max(std::fabs(h_output[i][j]), std::fabs(referenceData[i][j]));
            if (largest == 0){
                if (diff != 0){
                    std::cerr << "Result verification failed at element " << j << " of output " << i << "! Value: "<< h_output[i][j] << std::endl;
                    valid = false;
                }
            }

            T relative_error = (diff / largest) * 100;
            if (relative_error > tolerance_percent){
                std::cerr << "Result verification closeness failed at element " << j << " of output " << i << "! Value: "<< h_output[i][j] << " ref "<<referenceData[i][j] << std::endl;
                valid = false;
            }
        }
    }
    return valid;
}

template <typename T, typename Op>
bool verify_test
(
    std::string testName
  , uint numElements
  , uint os
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0]);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , const T cx
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cx);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , const T cx
  , const T cy
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cx, cy);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , uint os1
  , const T cx
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os, os1 });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], numElements, cx, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cfg.h_outputs[1], cx);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , uint os1
  , const T cx
  , const T cy
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os, os1 });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cfg.h_outputs[1], cx, cy);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , uint os1
  , uint os2
  , const T cx
  , const T cy
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os, os1, os2 });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], cfg.d_outputs[2], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cfg.h_outputs[1], cfg.h_outputs[2], cx, cy);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , uint os1
  , uint os2
  , const T cx
  , const T cy
  , const T cz
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os, os1, os2 });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], cfg.d_outputs[2], numElements, cx, cy, cz, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cfg.h_outputs[1], cfg.h_outputs[2], cx, cy, cz);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , uint os1
  , uint os2
  , uint os3
  , const T cx
  , const T cy
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os, os1, os2, os3 });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], cfg.d_outputs[2], cfg.d_outputs[3], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cfg.h_outputs[1], cfg.h_outputs[2], cfg.h_outputs[3], cx, cy);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , uint os1
  , uint os2
  , uint os3
  , const T cx
  , const T cy
  , const T cz
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os, os1, os2, os3 });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], cfg.d_outputs[2], cfg.d_outputs[3], numElements, cx, cy, cz, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cfg.h_outputs[1], cfg.h_outputs[2], cfg.h_outputs[3], cx, cy, cz);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T, typename Op>
bool verify_test
(
    const std::string& testName
  , uint numElements
  , uint os
  , uint os2
  , uint os3
  , uint os4
  , uint os5
  , const T cx
  , const T cy
  , Op op)
{    
    // Prepare the kernel launch
    auto cfg = prepareKernelLaunch<T>( testName, numElements, {}, { os, os2, os3, os4, os5 });

    if (!verifyCudaStatus("prepare", testName))
        return false;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], cfg.d_outputs[1], cfg.d_outputs[2], cfg.d_outputs[3], cfg.d_outputs[4], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", testName))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    op(0, cfg.h_outputs[0], cfg.h_outputs[1], cfg.h_outputs[2], cfg.h_outputs[3], cfg.h_outputs[4], cx, cy);
    std::vector<std::vector<T>> referenceData = defineReferenceData(cfg);
    if (!verifyEqualOutputs(h_out, referenceData))
       return false;
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

#endif  // BOOST_MATH_TEST_AUTODIFF_HPP
