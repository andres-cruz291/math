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

namespace diff = boost::math::differentiation;

template<typename T>
struct KernelLaunchConfig
{
    int numElements;
    std::vector<std::vector<T>> inputElements;
    std::vector<uint> outputElements;
    int threadsPerBlock;
    int blocksPerGrid;
    std::vector<T*> d_inputs;
    std::vector<T*> d_outputs;
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
    config.inputElements = inputVars;
    config.outputElements = outputVars;
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
        err = cudaMalloc(&config.d_outputs[i], outputVars[i] * sizeof(T));
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to allocate output " << i << std::endl;
            exit(EXIT_FAILURE);
        }
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
std::vector<std::vector<T>> returnOutputs(KernelLaunchConfig<T> cfg){
    std::vector<std::vector<T>> output(cfg.outputElements.size());
    for (uint i = 0; i < cfg.d_inputs.size(); ++i){
        cudaFree(cfg.d_inputs[i]);
    }
    for (uint i = 0; i < cfg.outputElements.size(); ++i){
        T h_output[cfg.outputElements[i]];
        output[i].resize(cfg.outputElements[i]);
        cudaMemcpy(h_output, cfg.d_outputs[i], cfg.outputElements[i] * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(cfg.d_outputs[i]);
        for (uint j = 0; j < cfg.outputElements[i]; ++j)
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
    bool err = true;
    for(uint i = 0; i < h_output.size(); ++i){
        for(uint j = 0; j < h_output[i].size(); ++j){
            if (h_output[i][j] != referenceData[i][j]){
                std::cerr << "Result verification failed at element " << j << " of output " << i << "! Value: "<< h_output[i][j] << std::endl;
                err = false;
            }
        }
    }
    return err;
}

#endif  // BOOST_MATH_TEST_AUTODIFF_HPP
