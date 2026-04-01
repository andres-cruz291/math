#ifndef BOOST_MATH_DIFFERENTIATION_AUTODIFF_STD_HPP
#define BOOST_MATH_DIFFERENTIATION_AUTODIFF_STD_HPP

#include <boost/math/tools/config.hpp>
#ifdef __CUDACC__
namespace std{
    
template <typename... Ts>
struct cuda_tuple;

template <>
struct cuda_tuple<> {};

template <typename T, typename... Ts>
struct cuda_tuple<T, Ts...> {
    T head;
    cuda_tuple<Ts...> tail;
};

template <typename... Ts>
BOOST_MATH_CUDA_ENABLED cuda_tuple<Ts...> make_tuple(Ts... xs)
{
    return { xs... };
}
} // namespace std
#endif
namespace boost {
namespace math {
namespace differentiation {
inline namespace autodiff_v1 {
namespace detail {
#ifdef __CUDACC__
template <typename Iterator, typename T>
BOOST_MATH_CUDA_ENABLED void fill
(
    Iterator first
  , Iterator last
  , const T& value)
{
    for (; first != last; ++first)
    {
        *first = value;
    }
}

template<typename Iterator1, typename Iterator2, typename T>
BOOST_MATH_CUDA_ENABLED T inner_product(Iterator1 first1, Iterator1 last1, Iterator2 first2, T init) {
  for (; first1 != last1; ++first1, ++first2) {
    init += (*first1) * (*first2);
  }
  return init;
}

template <typename Iterator1, typename T, typename BinaryOp>
BOOST_MATH_CUDA_ENABLED T accumulate(Iterator1 first, Iterator1 last, T init, BinaryOp op)
{
    for (; first != last; ++first)
    {
        init = op(init, *first);
    }
    return init;
}

template <class T, class U>
BOOST_MATH_CUDA_ENABLED constexpr auto min(const T& a, const U& b)
    -> typename std::common_type<T, U>::type
{
    using R = typename std::common_type<T, U>::type;
    return (b < a) ? static_cast<R>(b) : static_cast<R>(a);
}

template <class InputIt, class UnaryFunction>
BOOST_MATH_CUDA_ENABLED constexpr UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f)
{
    for (; first != last; ++first)
    {
        f(*first);
    }
    return f;
}

#else
using std::min;
using std::fill;
using std::for_each;

#endif
} // namespace detail
} // namespace autodiff_v1
} // namespace differentiation

namespace tools
{
#ifdef __CUDACC__
template <class T>
BOOST_MATH_CUDA_ENABLED constexpr const T& max(const T& a, const T& b)
{
    return (a < b) ? b : a;
}
#else
using std::max;
#endif
} // namespace tools
} // namespace math
} // namespace boost
#endif