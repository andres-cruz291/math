
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
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "test_autodiff_cuda.hpp"

using namespace boost::math::differentiation;
namespace bmp = boost::multiprecision;

/**
 * CUDA Kernels Device code
 *
 */

template <typename T, std::size_t m, std::size_t n>
struct ConstructorsOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2, T *out3, T *out4, T *out5) const {
        uint offset = m + 1;
        // Verify value-initialized instance has all 0 entries.
        const autodiff_fvar<T, m> empty = autodiff_fvar<T, m>();       
        fill_output_sv(empty, out, i, offset, offset);

        const auto empty2 = autodiff_fvar<T, m, n>();
        offset = (m + 1) * (n + 1);
        fill_output_dv(empty2, out2, i, offset, m + 1, n + 1);

        // Single variable
        const T cx = 10;
        const auto x = make_fvar<T, m>(cx);
        offset = m + 1;
        fill_output_sv(x, out3, i, offset, offset);

        const autodiff_fvar<T, n> xn = x;
        offset = n + 1;
        fill_output_sv(xn, out4, i, offset, offset);

        // Second independent variable
        const T cy = 100;
        const auto y = make_fvar<T, m, n>(cy);
        offset = (m + 1) * (n + 1);
        fill_output_dv(y, out5, i, offset, m + 1, n + 1);
    }
};

template <typename T>
BOOST_MATH_CUDA_ENABLED T uncast_return(const T& x) {
  return x == 0 ? 0 : 1;
}

template <typename T, std::size_t m>
struct ImplicitConstructorsOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const autodiff_fvar<T, m> x = 3;
        const autodiff_fvar<T, m> one = uncast_return(x);
        const autodiff_fvar<T, m> two_and_a_half = 2.5;
        const uint offset = 3;
        out[i * offset] = static_cast<T>(x);
        out[i * offset + 1] = static_cast<T>(one);
        out[i * offset + 2] = static_cast<T>(two_and_a_half);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct AssignmentOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1, T *out2) const {
        const T cx = 10;
        const T cy = 10;
        
        const uint offset = (m + 1) * (n + 1);
        
        autodiff_fvar<T, m, n> empty;
        // Single variable
        auto x = make_fvar<T, m>(cx);
        empty = static_cast<decltype(empty)>(x);        
        fill_output_dv(empty, out, i, offset, m + 1, n + 1);

        auto y = make_fvar<T, m, n>(cy);
        empty = y; // default assignment operator
        fill_output_dv(empty, out1, i, offset, m + 1, n + 1);

        empty = cx; // set a constant
        fill_output_dv(empty, out2, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct AdditionAssignmentOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1) const {
        const uint offset = (m + 1) * (n + 1);
        
        const T cx = 10;
        auto sum = autodiff_fvar<T, m, n>(); // zero-initialized
        // Single variable
        const auto x = make_fvar<T, m>(cx);
        sum += x;
        fill_output_dv(sum, out, i, offset, m + 1, n + 1);
        // Arithmetic constant
        const T cy = 11;
        sum = 0;
        sum += cy;
        fill_output_dv(sum, out1, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct SubtractionAssignmentOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1) const {
        const uint offset = (m + 1) * (n + 1);
        
        const T cx = 10;
        auto sum = autodiff_fvar<T, m, n>(); // zero-initialized
        // Single variable
        const auto x = make_fvar<T, m>(cx);
        sum -= x;
        fill_output_dv(sum, out, i, offset, m + 1, n + 1);
        // Arithmetic constant
        const T cy = 11;
        sum = 0;
        sum -= cy;
        fill_output_dv(sum, out1, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct MultiplicationAssignmentOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1) const {
        uint offset = (m + 1) * (n + 1);
        
        const T cx = 10;
        auto product = autodiff_fvar<T, m, n>(1); // unit-constant
        // Single variable
        const auto x = make_fvar<T, m>(cx);
        product *= x;
        fill_output_dv(product, out, i, offset, m + 1, n + 1);
        const T cy = 11;
        product = 1;
        product *= cy;
        fill_output_dv(product, out1, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m>
struct MultiplicationInfAssignmentOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        uint offset = (m + 1);
        auto x = make_fvar<T, m>(T(0.0));
        x *= std::numeric_limits<T>::infinity();
        fill_output_sv(x, out, i, offset, offset);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct DivisionAssignmentOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1) const {
        const uint offset = (m + 1) * (n + 1);

        const T cx = 16;
        auto quotient = autodiff_fvar<T, m, n>(1); // unit-constant
        // Single variable
        const auto x = make_fvar<T, m>(cx);
        quotient /= x;
        fill_output_dv(quotient, out, i, offset, m + 1, n + 1);
        // Arithmetic constant
        const T cy = 32;
        quotient = 1;
        quotient /= cx;
        fill_output_dv(quotient, out1, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct UnarySignsOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1) const {
        const T cx = 16;
        autodiff_fvar<T, m, n> lhs;
        const auto x = make_fvar<T, m>(cx);
        const uint offset = (m + 1) * (n + 1);
        
        lhs = static_cast<decltype(lhs)>(-x);
        fill_output_dv(lhs, out, i, offset, m + 1, n + 1);

        lhs = static_cast<decltype(lhs)>(+x);
        fill_output_dv(lhs, out1, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m>
 struct CastDoubleOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const T ca(13);
        const T j(12);
        const auto x = autodiff_fvar<T, m>(ca);
        const uint offset = 2;
        out[i * offset] = static_cast<T>(x);
        out[i * offset + 1] = static_cast<T>(j * x);
    }
};

template <typename T>
struct IntDoubleCastingOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const uint offset = 3;
        const T ca = 3;
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
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const uint offset = 3;
        const T ca = 3;
        const T cb = 4;
        const auto sum0 = autodiff_fvar<T, 0>(ca) + autodiff_fvar<T, 0>(cb);
        out[i * offset] = static_cast<T>(sum0);
        const auto sum1 = autodiff_fvar<T, 0>(ca) + cb;
        out[i * offset + 1] = static_cast<T>(sum1);
        const auto sum2 = ca + autodiff_fvar<T, 0>(cb);
        out[i * offset + 2] = static_cast<T>(sum2);
    }
};

template <typename T, std::size_t n>
struct Power8Op{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2) const {
        const uint offset = n + 1;
        
        const T ca = 3;
        auto x = make_fvar<T, n>(ca);
        // Test operator*=()
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

template <typename T, std::size_t m, std::size_t n>
struct Dim1MultiplicationOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2) const {
        const uint offset = n + 1;
        
        const T cy = 4;
        auto y0 = make_fvar<T, m>(cy);
        auto y = make_fvar<T, n>(cy);
        y *= y0;
        fill_output_sv(y, out1, i, offset, offset);
        
        y = y * cy;
        fill_output_sv(y, out2, i, offset, offset);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct Dim1And2MultiplicationOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const T cx = 3;
        const T cy = 4;
        auto x = make_fvar<T, m>(cx);
        auto y = make_fvar<T, m, n>(cy);
        y *= x;
        const uint offset = m * n;
        fill_output_dv(y, out, i, offset, m, n);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct Dim2AdditionOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2, T *out3) const {
        const T cx = 3;
        const auto x = make_fvar<T, m>(cx);
        uint offset = m + 1;
        fill_output_sv(x, out1, i, offset, offset);

        const T cy = 4;
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

template <typename T, std::size_t m, std::size_t n>
struct Dim2MultiplicationOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const T cx = 6;
        const auto x = make_fvar<T, m>(cx);
        const T cy = 5;
        const auto y = make_fvar<T, 0, n>(cy);
        const auto z = x * x * y * y * y;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct Dim2MultiplicationAndSubtractionOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const T cx = 6;
        const auto x = make_fvar<T, m>(cx);
        const T cy = 5;
        const auto y = make_fvar<T, 0, n>(cy);
        const auto z = x * x - y * y;
        const uint offset = (m + 1) * (n + 1);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);
    }
};

template <typename T, std::size_t m>
struct InverseOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out1, T *out2) const {
        const T cx = 4;
        const auto x = make_fvar<T, m>(cx);
        const auto xinv = x.inverse();
        const uint offset = (m + 1);
        fill_output_sv(xinv, out1, i, offset, offset);

        const auto zero = make_fvar<T, m>(0);
        const auto inf = zero.inverse();
        fill_output_sv(inf, out2, i, offset, offset);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct DivisionOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out1, T *out2, T *out3) const {
        const T cx = 16;
        auto x = make_fvar<T, m>(cx);
        const T cy = 4;
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

template <typename T, std::size_t m>
struct FabsTestOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2, T *out3, T *out4) const {
        using bmp::fabs;
        using detail::fabs;
        using std::fabs;
        
        const uint offset = (m + 1);
        
        const T cx = 11;
        const auto x = make_fvar<T, m>(cx);
        auto a = fabs(x);
        fill_output_sv(a, out, i, offset, offset);

        a = fabs(-x);
        fill_output_sv(a, out2, i, offset, offset);
        
        const auto xneg = make_fvar<T, m>(-cx);
        a = fabs(xneg);
        fill_output_sv(a, out3, i, offset, offset);

        const auto zero = make_fvar<T, m>(0);
        a = fabs(zero);
        fill_output_sv(a, out4, i, offset, offset);
    }
};

template <typename T, std::size_t m, std::size_t n>
struct EqualityOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T cx, T cy) const {
        const auto x = make_fvar<T, m>(cx);
        const auto y = make_fvar<T, 0, n>(cy);
        const uint offset = 2;
        out[ i * offset ] = static_cast<T>(x);
        out[ i * offset + 1 ] = static_cast<T>(y);
    }
};

template <typename T, std::size_t m>
struct CeilAndFloorOp {
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using bmp::ceil;
        using bmp::floor;
        using std::ceil;
        using std::floor;
        
        const uint testsSize = 3;
        T tests[]{T(-1.5), T(0.0), T(1.5)};
        const uint offset = (m + 1) * testsSize * 2;
        uint k = 0;
        for (T &test : tests) {
            const auto x = make_fvar<T, m>(test);
            auto c = ceil(x);
            auto f = floor(x);
            for (uint j = 0; j < (m + 1); ++j){
                out[i * offset + k * (m + 1) + j] = c.derivative(j);
                out[i * offset + ( m + 1 ) * testsSize + k * (m + 1) + j] = f.derivative(j);
            }
            k += 1;
        }
    }
};

/**
 * Host main routines
 */
template <typename T>
bool constructors(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    ConstructorsOp<T, m, n> op;
    return verify_test<T, ConstructorsOp<T, m, n>>("constructors", numElements, 
                       (m + 1), (m + 1) * (n + 1), (m + 1), (n + 1), (m + 1) * (n + 1), 
                       op );
}

template <typename T>
bool implicit_constructors(uint numElements)
{
    constexpr std::size_t m = 3;
    ImplicitConstructorsOp<T, m> op;
    return verify_test<T, ImplicitConstructorsOp<T, m>>(
        "implicit_constructors", numElements, 
        3,
        op 
    );
}

template <typename T>
bool assigment(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    AssignmentOp<T, m, n> op;
    return verify_test<T, AssignmentOp<T, m, n>>(
        "assigment", numElements, 
        (m + 1) * (n + 1), (m + 1) * (n + 1), (m + 1) * (n + 1),
        op 
    );
}

template <typename T>
bool addition_assignment(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    AdditionAssignmentOp<T, m, n> op;
    return verify_test<T, AdditionAssignmentOp<T, m, n>>(
        "addition_assignment", numElements, 
        (m + 1) * (n + 1), (m + 1) * (n + 1),
        op 
    );
}

template <typename T>
bool subtraction_assignment(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    SubtractionAssignmentOp<T, m, n> op;
    return verify_test<T, SubtractionAssignmentOp<T, m, n>>(
        "subtraction_assignment", numElements, 
        (m + 1) * (n + 1), (m + 1) * (n + 1),
        op 
    );
}

template <typename T>
bool multiplication_assignment(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    MultiplicationAssignmentOp<T, m, n> op;
    if (verify_test<T, MultiplicationAssignmentOp<T, m, n>>(
        "multiplication_assignment", numElements, 
        (m + 1) * (n + 1), (m + 1) * (n + 1), 
        op )
    ){
        MultiplicationInfAssignmentOp<T, m> op1;
        auto cfg = prepareKernelLaunch<T>( "multiplication_assignment", numElements, {}, { m + 1 });

        if (!verifyCudaStatus("prepare", "multiplication_assignment"))
            return false;
        
        watch w;
        // Execute the kernel
        apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, op1);
        cudaDeviceSynchronize();

        std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
        if (!verifyCudaStatus("launch", "multiplication_assignment"))
            return false;
        
        std::vector<std::vector<T>> h_out = returnOutputs(cfg);
        for (uint i = 0; i < numElements; ++i){
            for (uint j = 0; j < m + 1; ++j){
                if (j == 0){
                    if (!boost::math::isnan(h_out[0][ i * (m + 1) ])){
                        std::cerr << "Result verification failed at element " << i << " of multiplication_assignment_inf[0] Value: "<< h_out[0][ i * (m + 1) ] << std::endl;
                        return false;
                    }
                }else if(j == 1){
                    if (!boost::math::isinf(h_out[0][ i * (m + 1) + 1 ])){
                        std::cerr << "Result verification failed at element " << i << " of multiplication_assignment_inf[1] Value: "<< h_out[0][ i * (m + 1) + 1 ] << std::endl;
                        return false;
                    }
                } else{
                    if (h_out[0][ i * (m + 1) + j ] != 0){
                        std::cerr << "Result verification failed at element " << i << " of multiplication_assignment_inf[" << j << "] Value: "<< h_out[0][ i * (m + 1) + j ] << std::endl;
                        return false;
                    }
                }
            }
        }
        std::cout << "Test PASSED" << std::endl;
        std::cout << "Done\n";

        return true;
    }
    return false;
}

template <typename T>
bool division_assignment(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    DivisionAssignmentOp<T, m, n> op;
    return verify_test<T, DivisionAssignmentOp<T, m, n>>(
        "division_assignment", numElements, 
        (m + 1) * (n + 1), (m + 1) * (n + 1),
        op 
    );
}

template <typename T>
bool unary_signs(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    UnarySignsOp<T, m, n> op;
    return verify_test<T, UnarySignsOp<T, m, n>>(
        "unary_signs", numElements, 
        (m + 1) * (n + 1), (m + 1) * (n + 1),
        op 
    );
}

template <typename T>
bool cast_double(uint numElements)
{
    constexpr std::size_t m = 3;
    CastDoubleOp<T, m> op;
    return verify_test<T, CastDoubleOp<T, m>>(
        "cast_double", numElements, 
        2, 
        op 
    );
}

template <typename T>
bool int_double_casting(uint numElements)
{
    IntDoubleCastingOp<T> op;
    return verify_test<T, IntDoubleCastingOp<T>>(
        "int_double_casting", numElements, 
        3, 
        op 
    );
}

template <typename T>
bool scalar_addition(uint numElements)
{
    ScalarAdditionOp<T> op;
    return verify_test<T, ScalarAdditionOp<T>>(
        "scalar_addition", numElements, 
        3, 
        op 
    );
}

template <typename T>
bool power8(uint numElements)
{
    constexpr std::size_t n = 8u;
    Power8Op<T, n> op;
    return verify_test<T, Power8Op<T, n>>(
        "power8", numElements, 
        (n + 1), (n + 1), 
        op 
    );
}

template <typename T>
bool dim1_multiplication(uint numElements)
{
    constexpr std::size_t m = 2;
    constexpr std::size_t n = 3;
    Dim1MultiplicationOp<T, m, n> op;
    return verify_test<T, Dim1MultiplicationOp<T, m, n>>(
        "dim1_multiplication", numElements, 
        (n + 1), (n + 1), 
        op 
    );
}

template <typename T>
bool dim1and2_multiplication(uint numElements)
{
    constexpr std::size_t m = 2;
    constexpr std::size_t n = 3;
    Dim1And2MultiplicationOp<T, m, n> op;
    return verify_test<T, Dim1And2MultiplicationOp<T, m, n>>(
        "dim1and2_multiplication", numElements, 
        (m * n), 
        op 
    );
}

template <typename T>
bool dim2_addition(uint numElements)
{
    constexpr std::size_t m = 2;
    constexpr std::size_t n = 3;
    Dim2AdditionOp<T, m, n> op;
    return verify_test<T, Dim2AdditionOp<T, m, n>>(
        "dim2_addition", numElements, 
        (m + 1), (m * n), (4 * m),
        op 
    );
}

template <typename T>
bool dim2_multiplication(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    Dim2MultiplicationOp<T, m, n> op;
    return verify_test<T, Dim2MultiplicationOp<T, m, n>>(
        "dim2_multiplication", numElements, 
        (m + 1) * (n + 1), 
        op 
    );
}

template <typename T>
bool dim2_multiplication_and_subtraction(uint numElements)
{
    constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
    Dim2MultiplicationAndSubtractionOp<T, m, n> op;
    return verify_test<T, Dim2MultiplicationAndSubtractionOp<T, m, n>>(
        "dim2_multiplication_and_subtraction", numElements, 
        (m + 1) * (n + 1), 
        op 
    );
}

template <typename T>
bool inverse(uint numElements)
{
    constexpr std::size_t m = 3;
    InverseOp<T, m> op;
    return verify_test<T, InverseOp<T, m>>(
        "inverse", numElements, 
        (m + 1), (m + 1), 
        op 
    );
}

template <typename T>
bool division(uint numElements)
{    
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    DivisionOp<T, m, n> op;
    return verify_test<T, DivisionOp<T, m, n>>(
        "division", numElements, 
        (m + 1) * (n + 1), (m + 1), (m + 1) * (n + 1), (m + 1) * (n + 1),
        op 
    );
}

template <typename T>
bool equality(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    EqualityOp<T, m, n> op;
    auto cfg = prepareKernelLaunch<T>( "equality", numElements, {}, { 2 });

    if (!verifyCudaStatus("prepare", "equality"))
        return false;
    
    const T cx = 10;
    const T cy = 10;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", "equality"))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    for (uint i = 0; i < numElements; ++i){
        if (!(h_out[0][ i * 2 ] == h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [x == y] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] == cy)){
            std::cerr << "Result verification failed at element " << i << " [x == cy] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 + 1 ] == cx)){
            std::cerr << "Result verification failed at element " << i << " [cx == y] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] == cy)){
            std::cerr << "Result verification failed at element " << i << " [cy == x] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 + 1 ] == cx)){
            std::cerr << "Result verification failed at element " << i << " [y == cx] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
    }
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool inequality(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    EqualityOp<T, m, n> op;
    auto cfg = prepareKernelLaunch<T>( "inequality", numElements, {}, { 2 });

    if (!verifyCudaStatus("prepare", "inequality"))
        return false;
    
    const T cx = 10;
    const T cy = 11;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", "inequality"))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    for (uint i = 0; i < numElements; ++i){
        if (!(h_out[0][ i * 2 ] != h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [x != y] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] != cy)){
            std::cerr << "Result verification failed at element " << i << " [x != cy] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 + 1 ] != cx)){
            std::cerr << "Result verification failed at element " << i << " [cx != y] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] != cy)){
            std::cerr << "Result verification failed at element " << i << " [cy != x] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 + 1 ] != cx)){
            std::cerr << "Result verification failed at element " << i << " [y != cx] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
    }
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool less_than_or_equal_to(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    EqualityOp<T, m, n> op;
    auto cfg = prepareKernelLaunch<T>( "less_than_or_equal_to", numElements, {}, { 2 });

    if (!verifyCudaStatus("prepare", "less_than_or_equal_to"))
        return false;
    
    const T cx = 10;
    const T cy = 11;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", "less_than_or_equal_to"))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    for (uint i = 0; i < numElements; ++i){
        if (!(h_out[0][ i * 2 ] <= h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [x <= y] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] <= h_out[0][ i * 2 + 1 ] - 1)){
            std::cerr << "Result verification failed at element " << i << " [x <= y - 1] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] < h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [x < y] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] <= cy)){
            std::cerr << "Result verification failed at element " << i << " [x <= cy] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] <= cy - 1)){
            std::cerr << "Result verification failed at element " << i << " [x <= cy - 1] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] < cy)){
            std::cerr << "Result verification failed at element " << i << " [x < cy] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(cx <= h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [cx <= y] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(cx <= h_out[0][ i * 2 + 1 ] - 1)){
            std::cerr << "Result verification failed at element " << i << " [cx <= y - 1] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(cx < h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [cx < y] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
    }
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool greater_than_or_equal_to(uint numElements)
{
    constexpr std::size_t m = 3;
    constexpr std::size_t n = 4;
    EqualityOp<T, m, n> op;
    auto cfg = prepareKernelLaunch<T>( "greater_than_or_equal_to", numElements, {}, { 2 });

    if (!verifyCudaStatus("prepare", "greater_than_or_equal_to"))
        return false;
    
    const T cx = 11;
    const T cy = 10;
    
    watch w;
    // Execute the kernel
    apply_op<T><<<cfg.blocksPerGrid, cfg.threadsPerBlock>>>(cfg.d_outputs[0], numElements, cx, cy, op);
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel done in: " << w.elapsed() << "s" << std::endl;
    if (!verifyCudaStatus("launch", "greater_than_or_equal_to"))
        return false;
    
    std::vector<std::vector<T>> h_out = returnOutputs(cfg);
    for (uint i = 0; i < numElements; ++i){
        if (!(h_out[0][ i * 2 ] >= h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [x >= y] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] >= h_out[0][ i * 2 + 1 ] + 1)){
            std::cerr << "Result verification failed at element " << i << " [x >= y + 1] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] > h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [x > y] x: "<< h_out[0][ i * 2 ] << " y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] >= cy)){
            std::cerr << "Result verification failed at element " << i << " [x >= cy] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] >= cy + 1)){
            std::cerr << "Result verification failed at element " << i << " [x >= cy + 1] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(h_out[0][ i * 2 ] > cy)){
            std::cerr << "Result verification failed at element " << i << " [x > cy] x: "<< h_out[0][ i * 2 ] << std::endl;
            return false;
        }
        if (!(cx >= h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [cx >= y] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(cx >= h_out[0][ i * 2 + 1 ] + 1)){
            std::cerr << "Result verification failed at element " << i << " [cx >= y + 1] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
        if (!(cx > h_out[0][ i * 2 + 1 ])){
            std::cerr << "Result verification failed at element " << i << " [cx > y] y: "<< h_out[0][ i * 2 + 1 ] << std::endl;
            return false;
        }
    }
    std::cout << "Test PASSED" << std::endl;
    std::cout << "Done\n";

    return true;
}

template <typename T>
bool fabs_test(uint numElements)
{    
    constexpr std::size_t m = 3;
    FabsTestOp<T, m> op;
    return verify_test<T, FabsTestOp<T, m>>(
        "fabs_test", numElements, 
        (m + 1), (m + 1), (m + 1), (m + 1), 
        op 
    );
}

template <typename T>
bool ceil_and_floor(uint numElements)
{    
    constexpr std::size_t m = 3;
    CeilAndFloorOp<T, m> op;
    return verify_test<T, CeilAndFloorOp<T, m>>(
        "ceil_and_floor", numElements, 
        2 * m * (m + 1), 
        op 
    );
}

template <typename float_type>
bool main_tests_1(uint numElements){
    bool all_passed = true;
    if (!constructors<float_type>(numElements))
        all_passed = false;
    if (!implicit_constructors<float_type>(numElements))
        all_passed = false;
    if (!assigment<float_type>(numElements))
        all_passed = false;
    if (!addition_assignment<float_type>(numElements))
        all_passed = false;
    if (!subtraction_assignment<float_type>(numElements))
        all_passed = false;
    if (!multiplication_assignment<float_type>(numElements))
        all_passed = false;
    if (!division_assignment<float_type>(numElements))
        all_passed = false;
    if (!unary_signs<float_type>(numElements))
        all_passed = false;
    if (!cast_double<float_type>(numElements))
        all_passed = false;
    if (!int_double_casting<float_type>(numElements))
        all_passed = false;
    if (!scalar_addition<float_type>(numElements))
        all_passed = false;
    if (!power8<float_type>(numElements))
        all_passed = false;
    if (!dim1_multiplication<float_type>(numElements))
        all_passed = false;
    if (!dim1and2_multiplication<float_type>(numElements))
        all_passed = false;
    if (!dim2_addition<float_type>(numElements))
        all_passed = false;
    if (!dim2_multiplication<float_type>(numElements))
        all_passed = false;
    if (!dim2_multiplication_and_subtraction<float_type>(numElements))
        all_passed = false;
    if (!inverse<float_type>(numElements))
        all_passed = false;
    if (!division<float_type>(numElements))
        all_passed = false;
    if (!equality<float_type>(numElements))
        all_passed = false;
    if (!inequality<float_type>(numElements))
        all_passed = false;
    if (!less_than_or_equal_to<float_type>(numElements))
        all_passed = false;
    if (!greater_than_or_equal_to<float_type>(numElements))
        all_passed = false;
    if (!fabs_test<float_type>(numElements))
        all_passed = false;
    if (!ceil_and_floor<float_type>(numElements))
        all_passed = false;
        
    return all_passed;
}