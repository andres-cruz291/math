
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
 struct CastDoubleOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T ca, T j) const {
        const auto x = autodiff_fvar<T, m>(ca);
        const uint offset = 2;
        out[i * offset] = static_cast<T>(x);
        out[i * offset + 1] = static_cast<T>(j * x);
    }
};

template <typename T>
struct IntDoubleCastingOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T ca) const {
        const auto x = autodiff_fvar<T, m>(ca);
        const uint offset = 3;
        const auto x0 = make_fvar<T, 0>(ca);
        out[i * offset] = static_cast<T>(x0);
        const auto x1 = make_fvar<T, 1>(ca);
        out[i * offset + 1] = static_cast<T>(x1);
        const auto x2 = make_fvar<T, 2>(ca);
        out[i * offset + 2] = static_cast<T>(x2);
    }
};

template <typename T>
struct ScalarAdditionOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T ca, T cb) const {
        const auto x = autodiff_fvar<T, m>(ca);
        const uint offset = 3;
        const auto sum0 = autodiff_fvar<T, 0>(ca) + autodiff_fvar<T, 0>(cb);
        out[i * offset] = static_cast<T>(sum0);
        const auto sum1 = autodiff_fvar<T, 0>(ca) + cb;
        out[i * offset + 1] = static_cast<T>(sum1);
        const auto sum2 = ca + autodiff_fvar<T, 0>(cb);
        out[i * offset + 2] = static_cast<T>(sum2);
    }
};

template <typename T, int n>
struct Power8Op{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2, T ca) const {
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
};

template <typename T, int m, int n>
struct Dim1MultiplicationOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2, T cy) const {
        const uint offset = n + 1;
        
        auto y0 = make_fvar<T, m>(cy);
        auto y = make_fvar<T, n>(cy);
        y *= y0;
        fill_output_sv(y, out1, i, offset, offset);
        
        y = y * cy;
        fill_output_sv(y, out2, i, offset, offset);
    }
};

template <typename T, int m, int n>
struct Dim1And2MultiplicationOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T cx, T cy) const {
        auto x = make_fvar<T, m>(cx);
        auto y = make_fvar<T, m, n>(cy);
        y *= x;
        const uint offset = m * n;
        fill_output_dv(y, out, i, offset, m, n);
    }
};

template <typename T, int m, int n>
struct Dim2AdditionOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2, T *out3, T cx, T cy) const {
        const auto x = make_fvar<T, m>(cx);
        uint offset = m + 1;
        fill_output_sv(x, out1, i, offset, offset);

        const auto y = make_fvar<T, m, n>(cy);
        offset = (m * n);
        out2[i * offset] = static_cast<T>(y.derivative(0));
        out2[i * offset + 1] = static_cast<T>(y.derivative(1));
        for (uint j = 0; j < m; ++j)
            for (uint k = 0; k < m; ++k)
                out2[i * offset + 2 + (j * m) + k] = y.derivative(j, k);

        const auto z = x + y;
        offset = 4 * m;
        for (uint j = 0; j < m; ++j)
            for (uint k = 0; k < m; ++k){
                out3[i * offset + (j * m) + k] = z.derivative(j, k);
                out3[i * offset + (2 * m) + (j * m) + k] = z.derivative(j).derivative(k);
            }
    }
};

template <typename T>
struct Dim2MultiplicationOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T cx, T cy) const {
        const auto x = make_fvar<T, m>(cx);
        const auto y = make_fvar<T, 0, n>(cy);
        const auto z = x * x * y * y * y;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);
    }
};

template <typename T>
struct Dim2MultiplicationAndSubtractionOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T cx, T cy) const {
        const auto x = make_fvar<T, m>(cx);
        const auto y = make_fvar<T, 0, n>(cy);
        const auto z = x * x - y * y;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);
    }
};

template <typename T>
struct InverseOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2, T cx) const {
        const auto x = make_fvar<T, m>(cx);
        const auto xinv = x.inverse();
        const uint offset = (m + 1);
        fill_output_sv(xinv, out1, i, offset, offset);

        const auto zero = make_fvar<T, m>(0);
        const auto inf = zero.inverse();
        fill_output_sv(inf, out2, i, offset, offset);
    }
};

template <typename T>
struct DivisionOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1, T *out2, T *out3, T cx, T cy) const {
        auto x = make_fvar<T, m>(cx);
        auto y = make_fvar<T, 1, n>(cy);
        auto z = x * x / (y * y);
        uint offset = (m + 1) * (n + 1);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);

        auto x1 = make_fvar<T, m>(cx);
        auto z1 = x1 / cy;
        offset = (m + 1);
        fill_output_sv(z1, out1, i, offset, offset);
        
        auto y2 = make_fvar<T, m, n>(cy);
        auto z2 = cx / y2;
        offset = (m + 1)* (n + 1);
        fill_output_dv(z2, out2, i, offset, m + 1, n + 1);

        const auto z3 = y / x;
        offset = (m + 1)* (n + 1);
        fill_output_dv(z3, out3, i, offset, m + 1, n + 1);
    }
};

/**
 * Host main routines
 */
template <typename T>
bool cast_double(uint numElements)
{
    CastDoubleOp<T> op;
    return verify_test("cast_double", numElements, 
                       2, 
                       T(13), T(12),
                       op );
}

template <typename T>
bool int_double_casting(uint numElements)
{
    IntDoubleCastingOp<T> op;
    return verify_test("int_double_casting", numElements, 
                       3, 
                       T(3),
                       op );
}

template <typename T>
bool scalar_addition(uint numElements)
{
    ScalarAdditionOp<T> op;
    return verify_test("scalar_addition", numElements, 
                       3, 
                       T(3), T(4), 
                       op );
}

template <typename T>
bool power8(uint numElements)
{
    constexpr std::size_t n = 8u;
    Power8Op<T, n> op;
    return verify_test("power8", numElements, 
                       (n + 1), (n + 1), 
                       T(3),
                       op );
}

template <typename T>
bool dim1_multiplication(uint numElements)
{
    constexpr std::size_t m = 2;
    constexpr std::size_t n = 3;
    Dim1MultiplicationOp<T, m, n> op;
    return verify_test("dim1_multiplication", numElements, 
                       (n + 1), (n + 1), 
                       T(4),
                       op );
}

template <typename T>
bool dim1and2_multiplication(uint numElements)
{
    constexpr std::size_t m = 2;
    constexpr std::size_t n = 3;
    Dim1And2MultiplicationOp<T, m, n> op;
    return verify_test("dim1and2_multiplication", numElements, 
                       (m * n), 
                       T(3), T(4),
                       op );
}

template <typename T>
bool dim2_addition(uint numElements)
{
    constexpr std::size_t m = 2;
    constexpr std::size_t n = 3;
    Dim2AdditionOp<T, m, n> op;
    return verify_test("dim2_addition", numElements, 
                       (m + 1), (m * n), (4 * m),
                       T(3), T(4),
                       op );
}

template <typename T>
bool dim2_multiplication(uint numElements)
{
    Dim2MultiplicationOp<T> op;
    return verify_test("dim2_multiplication", numElements, 
                       (m + 1) * (n + 1), 
                       T(6), T(5),
                       op );
}

template <typename T>
bool dim2_multiplication_and_subtraction(uint numElements)
{
    Dim2MultiplicationAndSubtractionOp<T> op;
    return verify_test("dim2_multiplication_and_subtraction", numElements, 
                       (m + 1) * (n + 1), 
                       T(6), T(5),
                       op );
}

template <typename T>
bool inverse(uint numElements)
{
    InverseOp<T> op;
    return verify_test("inverse", numElements, 
                       (m + 1), (m + 1), 
                       T(4),
                       op );
}

template <typename T>
bool division(uint numElements)
{    
    DivisionOp<T> op;
    return verify_test("division", numElements, 
                       (m + 1) * (n + 1), (m + 1), (m + 1) * (n + 1), (m + 1) * (n + 1),
                       T(16), T(4),
                       op );
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
    if (!dim1and2_multiplication<float_type>(numElements))
        return false;
    if (!dim2_addition<float_type>(numElements))
        return false;
    if (!dim2_multiplication<float_type>(numElements))
        return false;
    if (!dim2_multiplication_and_subtraction<float_type>(numElements))
        return false;
    if (!inverse<float_type>(numElements))
        return false;
    if (!division<float_type>(numElements))
        return false;

    return true;
}