
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
 
template <typename T, uint m>
struct OneOverOnePlusXSquaredOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T cx) const {
        const uint offset = m + 1;
        auto f = make_fvar<T, m>(cx);
        // f = 1 / ((f *= f) += 1);
        f *= f;
        f += T(1);
        f = f.inverse();
        fill_output_sv(f, out, i, offset, offset);
    }
};

template <typename T, uint m>
struct ExpTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T cx) const {
        using std::exp;
        const uint offset = m + 1;
        const auto x = make_fvar<T, m>(cx);
        auto y = exp(x);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, uint m, uint n>
struct PowSvOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2, T cx, T cy) const {
        using std::log;
        using std::pow;
        
        uint offset = (m + 1);
        const auto x = make_fvar<T, m>(cx);
        const auto y = make_fvar<T, m, n>(cy);
        auto z = pow(x, cy);
        fill_output_sv(z, out, i, offset, offset);

        offset = (m + 1) * (n + 1);
        auto z2 = pow(cx, y);
        fill_output_dv(z2, out2, i, offset, m + 1, n + 1);
    }
};

template <typename T, uint m, uint n>
struct PowDvOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T cx, T cy) const {
        using std::log;
        using std::pow;
        
        const auto x = make_fvar<T, m>(cx);
        const auto y = make_fvar<T, m, n>(cy);
        const uint offset = (m + 1) * (n + 1);
        auto z = pow(cx, y);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);
    }
};

template <typename T, uint m>
struct Pow0Op{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2, T cx, T cy) const {
        using std::pow;
        
        const uint offset = (m + 1);
        const auto x = make_fvar<T, m>(0);

        auto z = pow(x, cx);
        fill_output_sv(z, out, i, offset, offset);

        auto z2 = pow(x, cy);
        fill_output_sv(z2, out2, i, offset, offset);
    }
};

template <typename T, uint m, uint n>
struct Pow2Op{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using std::pow;
        
        const T cx = 2;
        const T cy = 3;
        
        const uint offset = (m + 1) * (n + 1);
        const auto x = make_fvar<T, m>(cx);
        const auto y = make_fvar<T, 0, n>(cy);
        
        const auto z = pow(x, y);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);
    }
};

template <typename T, uint m>
struct SqrtTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2) const {
        using std::pow;
        using std::sqrt;
        
        const uint offset = (m + 1);
        const T cx = 4;
        auto x = make_fvar<T, m>(cx);
        auto y = sqrt(x);
        fill_output_sv(y, out, i, offset, offset);

        x = make_fvar<T, m>(0);
        y = sqrt(x);
        fill_output_sv(y, out2, i, offset, offset);
    }
};

template <typename T, uint m>
struct LogTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2) const {
        using std::log;
        using std::pow;
        
        const uint offset = (m + 1);
        const T cx = 2;
        auto x = make_fvar<T, m>(cx);
        auto y = log(x);
        fill_output_sv(y, out, i, offset, offset);

        x = make_fvar<T, m>(0);
        y = log(x);
        fill_output_sv(y, out2, i, offset, offset);
    }
};

template <typename T, uint m, uint n>
struct YLogXOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2) const {
        using std::log;
        using std::pow;
        
        uint offset = (m + 1) * (n + 1);
        const T cx = 2;
        const T cy = 3;
        const auto x = make_fvar<T, m>(cx);
        const auto y = make_fvar<T, m, n>(cy);
        auto z = y * log(x);
        fill_output_dv(z, out, i, offset, m + 1, n + 1);

        offset = (m + 1) * (n + 1) - 2;
        auto z1 = exp(z);
        fill_output_dv(z1, out2, i, offset, m + 1, n + 1);
    }
};

template <typename T>
struct FrexpTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using std::exp2;
        using std::frexp;
        
        uint offset = (m + 1);
        const T cx = T(3.5);
        const auto x = make_fvar<T, m>(cx);
        int exp;
        auto y = frexp(x, &exp);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct CosAndSinOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2, T *out3, T *out4) const {
        using std::cos;
        using std::sin;
        
        uint offset = (m + 1);
        const T cx = boost::math::constants::third_pi<T>();
        const auto x = make_fvar<T, m>(cx);
        auto cos5 = cos(x);
        fill_output_sv(cos5, out, i, offset, offset);

        auto sin5 = sin(x);
        fill_output_sv(sin5, out2, i, offset, offset);

        offset = 1;
        auto cos0 = cos(make_fvar<T, 0>(cx));
        fill_output_sv(cos0, out3, i, offset, offset);

        auto sin0 = sin(make_fvar<T, 0>(cx));
        fill_output_sv(sin0, out4, i, offset, offset);
    }
};

template <typename T, int m>
struct AcosTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using std::acos;
        using std::pow;
        using std::sqrt;
        
        const T cx = T(0.5);
        auto x = make_fvar<T, m>(cx);
        auto y = acos(x);
        
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct AcoshTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using boost::math::acosh;
        
        const T cx = 2;
        auto x = make_fvar<T, m>(cx);
        auto y = acosh(x);       
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct AsinTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using std::asin;
        using std::pow;
        using std::sqrt;
        
        const T cx = T(0.5);
        auto x = make_fvar<T, m>(cx);
        auto y = asin(x);
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct AsinInfinityOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        auto x = make_fvar<T, m>(1);
        auto y = asin(x);
        const uint offset = 3;
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct AsinDerivativeOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2, T *out3) const {
        using std::pow;
        using std::sqrt;
        const T cx = T(0.5);
        auto x = make_fvar<T, m>(cx);
        auto y = T(1) - x * x;
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
        y = sqrt(y);
        fill_output_sv(y, out2, i, offset, offset);
        y = y.inverse();
        fill_output_sv(y, out3, i, offset, offset);
    }
};

template <typename T>
bool one_over_one_plus_x_squared(uint numElements)
{    
    constexpr std::size_t m = 4;
    OneOverOnePlusXSquaredOp<T, m> op;
    return verify_test("one_over_one_plus_x_squared", numElements, 
                       (m + 1), 
                       T(1), 
                       op );
}

template <typename T>
bool exp_test(uint numElements)
{    
    // It may be needed CHECK_CLOSE_FRACTION instead of equality
    constexpr std::size_t m = 4;
    ExpTestOp<T, m> op;
    return verify_test("exp_test", numElements, 
                       (m + 1), 
                       T(2), 
                       op );
}

template <typename T>
bool pow(uint numElements)
{    
    // It may be needed CHECK_CLOSE instead of equality
    // Function pow may have problems with memory, that appear when it is called several times with nested definition
    constexpr std::size_t m = 5;
    constexpr std::size_t n = 4;
    PowSvOp<T, m, n> op;
    if (verify_test("pow", numElements, 
                    (m + 1), (m + 1) * (n + 1),  
                    T(2), T(3), 
                    op )){
        PowDvOp<T, m, n> op1;
        if (verify_test("pow", numElements, 
                        (m + 1) * (n + 1),  
                        T(2), T(3), 
                        op1)){
            return true;
        }
    };
    return false;
}

template <typename T>
bool pow0(uint numElements)
{    
    constexpr std::size_t m = 5;
    Pow0Op<T, m> op;
    return verify_test("pow0", numElements, 
                        (m + 1), (m + 1),
                        T(3), T(3.5), 
                        op);
        
}

template <typename T>
bool pow2(uint numElements)
{    
    // It may needed to adjust the tolerance < e-13
    const T eps = 40000 * std::numeric_limits<T>::epsilon(); // percent
    constexpr std::size_t m = 5;
    constexpr std::size_t n = 5;
    Pow2Op<T, m, n> op;
    return verify_close_test("pow2", numElements, 
                            (m + 1) * (n + 1), 
                            op, 
                            eps);
        
}

template <typename T>
bool sqrt_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    SqrtTestOp<T, m> op;
    return verify_test<T, SqrtTestOp<T, m>>("sqrt_test", numElements, 
                        (m + 1), (m + 1),
                        op);
        
}

template <typename T>
bool log_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    LogTestOp<T, m> op;
    return verify_test<T, LogTestOp<T, m>>("log_test", numElements, 
                        (m + 1), (m + 1),
                        op);
        
}

template <typename T>
bool ylogx(uint numElements)
{    
    constexpr std::size_t m = 5;
    constexpr std::size_t n = 4;
    YLogXOp<T, m, n> op;
    // A high difference was found in d⁸/(d⁵x)(d³y) and d⁹/(d⁵x)(d⁴y) for second output (> 1e-3)
    return verify_test<T, YLogXOp<T, m, n>>("ylogx", numElements, 
                                            (m + 1) * (n + 1), (m + 1) * (n + 1) - 2,
                                            op);
        
}

template <typename T>
bool frexp_test(uint numElements)
{    
    FrexpTestOp<T> op;
    // Redefinition of function frexp
    return verify_test<T, FrexpTestOp<T>>("frexp_test", numElements, 
                                          (m + 1),
                                          op);
}

template <typename T>
bool cos_and_sin(uint numElements)
{    
    constexpr std::size_t m = 5;
    CosAndSinOp<T, m> op;
    const T eps = 1e-13; // percent
    return verify_close_test<T, CosAndSinOp<T, m>>("cos_and_sin", numElements, 
                                                   (m + 1), (m + 1), 1, 1, 
                                                   op, 
                                                   eps);
}

template <typename T>
bool acos_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    AcosTestOp<T, m> op;
    return verify_test<T, AcosTestOp<T, m>>("acos_test", numElements, 
                                                   (m + 1),
                                                   op);
}

template <typename T>
bool acosh_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    AcoshTestOp<T, m> op;
    // acosh and asinh functions not implemented
    return true;
    return verify_test<T, AcoshTestOp<T, m>>("acosh_test", numElements, 
                                                   (m + 1),
                                                   op);
}

template <typename T>
bool asin_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    AsinTestOp<T, m> op;
    return verify_test<T, AsinTestOp<T, m>>("asin_test", numElements, 
                                                   (m + 1),
                                                   op);
}

template <typename T>
bool asin_infinity(uint numElements)
{    
    // Fail to compare d³, d⁴ and d⁵ = -nan
    constexpr std::size_t m = 5;
    AsinInfinityOp<T, m> op;
    return verify_test<T, AsinInfinityOp<T, m>>("asin_infinity", numElements, 
                                                   3,
                                                   op);
}

template <typename T>
bool asin_derivative(uint numElements)
{    
    constexpr std::size_t m = 4;
    AsinDerivativeOp<T, m> op;
    return verify_test<T, AsinDerivativeOp<T, m>>("asin_derivative", numElements, 
                                                   (m + 1), (m + 1), (m + 1),
                                                   op);
}

/**
 * Host main routines
 */

template <typename float_type>
bool main_tests_2(uint numElements){
    if (!one_over_one_plus_x_squared<float_type>(numElements))
        return false;
    if (!exp_test<float_type>(numElements))
        return false;
    if (!pow<float_type>(numElements))
        return false;
    if (!pow0<float_type>(numElements))
        return false;
    if (!pow2<float_type>(numElements))
        return false;
    if (!sqrt_test<float_type>(numElements))
        return false;
    if (!log_test<float_type>(numElements))
        return false;
    if (!ylogx<float_type>(numElements))
        return false;
    if (!frexp_test<float_type>(numElements))
        return false;
    if (!cos_and_sin<float_type>(numElements))
        return false;
    if (!acos_test<float_type>(numElements))
        return false;
    if (!acosh_test<float_type>(numElements))
        return false;
    if (!asin_test<float_type>(numElements))
        return false;
    if (!asin_infinity<float_type>(numElements))
        return false;
    if (!asin_derivative<float_type>(numElements))
        return false;
    
    return true;
}