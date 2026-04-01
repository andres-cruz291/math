
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
namespace bmp = boost::multiprecision;

/**
 * CUDA Kernels Device code
 *
 */

template <typename T, int m>
struct AtanhTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using boost::math::atanh;
        
        const T cx = T(0.5);
        auto x = make_fvar<T, m>(cx);
        auto y = atanh(x);
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct AtanTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using namespace boost;
        
        const T cx = 1;
        const auto x = make_fvar<T, m>(cx);
        auto y = atan(x);
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct ErftestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using namespace boost;
        using boost::math::erf;
        
        const T cx = 1;
        const auto x = make_fvar<T, m>(cx);
        auto y = erf(x);
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T, int m>
struct SinctestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2) const {
        using namespace boost;
        using boost::math::erf;
        
        const T cx = 1;
        auto x = make_fvar<T, m>(cx);
        auto y = sinc(x);
        uint offset = m;
        fill_output_sv(y, out, i, offset, offset);

        auto y2 = sinc(make_fvar<T, 10>(0));
        offset = 11;
        fill_output_sv(y2, out2, i, offset, offset);
    }
};

template <typename T, int m>
struct SinhAndCoshOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out, T *out2) const {
        using namespace boost;
        using boost::math::erf;
        
        const T cx = 1;
        auto x = make_fvar<T, m>(cx);
        auto s = sinh(x);
        
        const uint offset = (m + 1);
        fill_output_sv(s, out, i, offset, offset);

        auto c = cosh(x);
        fill_output_sv(c, out2, i, offset, offset);
    }
};

template <typename T, int m>
struct TanhTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        using bmp::fabs;
        using bmp::tanh;
        using detail::fabs;
        using detail::tanh;
        using std::fabs;
        using std::tanh;
        
        const T cx = 1;
        auto x = make_fvar<T, m>(cx);
        auto t = tanh(x);
        
        const uint offset = (m + 1);
        fill_output_sv(t, out, i, offset, offset);
    }
};

template <typename T, int m>
struct TanTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const T cx = boost::math::constants::third_pi<T>();
        const auto x = make_fvar<T, m>(cx);
        auto y = tan(x);
        
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T>
struct FmodTestOp{
    BOOST_MATH_CUDA_ENABLED void operator()(uint i, T *out) const {
        const T cx = T(3.25);
        const T cy = T(0.5);
        auto x = make_fvar<T, m>(cx);
        auto y = fmod(x, autodiff_fvar<T, m>(cy));
        
        const uint offset = (m + 1);
        fill_output_sv(y, out, i, offset, offset);
    }
};

template <typename T>
bool atanh_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    AtanhTestOp<T, m> op;
    return verify_test<T, AtanhTestOp<T, m>>("atanh_test", numElements, 
                                                   (m + 1),
                                                   op);
}

template <typename T>
bool atan_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    AtanTestOp<T, m> op;
    return verify_test<T, AtanTestOp<T, m>>("atan_test", numElements, 
                                                   (m + 1),
                                                   op);
}

template <typename T>
bool erf_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    ErftestOp<T, m> op;
    const T eps = 300 * 100 * boost::math::tools::epsilon<T>();
    return verify_close_test<T, ErftestOp<T, m>>("erf_test", numElements, 
                                                   (m + 1),
                                                   op, 
                                                   eps);
}

template <typename T>
bool sinc_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    SinctestOp<T, m> op;
    const T eps = 20000 * boost::math::tools::epsilon<T>(); // percent
    // High difference in the 5th derivative of first output (> 1e-3) for float
    return verify_close_test<T, SinctestOp<T, m>>("sinc_test", numElements, 
                                                   m, 11, 
                                                   op,
                                                   eps);
}

template <typename T>
bool sinh_and_cosh(uint numElements)
{    
    constexpr std::size_t m = 5;
    SinhAndCoshOp<T, m> op;
    const T eps = 300 * boost::math::tools::epsilon<T>();
    return verify_close_test<T, SinhAndCoshOp<T, m>>("sinh_and_cosh", numElements, 
                                                   (m + 1), (m + 1),
                                                   op,
                                                   eps);
}

template <typename T>
bool tanh_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    TanhTestOp<T, m> op;
    const T eps = 10000 * boost::math::tools::epsilon<T>();
    return verify_close_test<T, TanhTestOp<T, m>>("tanh_test", numElements, 
                                                   (m + 1), 
                                                   op,
                                                   eps);
}

template <typename T>
bool tan_test(uint numElements)
{    
    constexpr std::size_t m = 5;
    TanTestOp<T, m> op;
    const T eps = 800 * boost::math::tools::epsilon<T>();
    return verify_close_test<T, TanTestOp<T, m>>("tan_test", numElements, 
                                                   (m + 1), 
                                                   op,
                                                   eps);
}

template <typename T>
bool fmod_test(uint numElements)
{    
    FmodTestOp<T> op;
    return verify_test<T, FmodTestOp<T>>("fmod_test", numElements, 
                                                   (m + 1), 
                                                   op);
}

template <typename float_type>
bool main_tests_3(uint numElements){
    if (!atanh_test<float_type>(numElements))
        return false;
    if (!atan_test<float_type>(numElements))
        return false;
    if (!erf_test<float_type>(numElements))
        return false;
    if (!sinc_test<float_type>(numElements))
        return false;
    if (!sinh_and_cosh<float_type>(numElements))
        return false;
    if (!tanh_test<float_type>(numElements))
        return false;
    if (!tan_test<float_type>(numElements))
        return false;
    if (!fmod_test<float_type>(numElements))
        return false;
            
    return true;
}

//std::cout << "eps "<<eps << std::endl;