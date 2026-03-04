/*
 * Copyright © 2025  Andres Cruz
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General
 * Public License along with this program. If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef BOOST_MATH_DIFFERENTIATION_ARRAY_HPP
#define BOOST_MATH_DIFFERENTIATION_ARRAY_HPP

#include <boost/math/tools/config.hpp>

#ifndef __CUDACC__
# include <array>
#endif

namespace boost {
namespace math {
namespace differentiation {

#ifndef __CUDACC__

template <typename T, size_t N>
struct std_array
  : std::array<T, N> {
  public:
      enum { static_size = N };
};

#else /* __CUDACC__ */

//
// The purpose of a std_array is to serve as the underlying
// array type to a fixed-length algebraic vector. It defines
// operator[] to allow convenient access of its components.
//
template <typename T, size_t N>
struct std_array
{
    struct forward_iterator
    {
        T* ptr;

        BOOST_MATH_CUDA_ENABLED forward_iterator(T* p) : ptr(p) {}
        
        BOOST_MATH_CUDA_ENABLED T& operator*() const { return *ptr; }

        BOOST_MATH_CUDA_ENABLED forward_iterator& operator++()
        {
            ++ptr;
            return *this;
        }

        BOOST_MATH_CUDA_ENABLED forward_iterator operator+(size_t x) const {
            return forward_iterator{ ptr + x };
        }

        BOOST_MATH_CUDA_ENABLED forward_iterator& operator+=(size_t x) {
            ptr += x;
            return *this;
        }

        BOOST_MATH_CUDA_ENABLED bool operator!=(const forward_iterator& other) const
        {
            return ptr != other.ptr;
        }
    };

    struct reverse_iterator {
        T const* ptr;

        BOOST_MATH_CUDA_ENABLED T const& operator*() const { return *ptr; }

        BOOST_MATH_CUDA_ENABLED reverse_iterator& operator++() {
            --ptr;
            return *this;
        }

        BOOST_MATH_CUDA_ENABLED reverse_iterator operator+(size_t x) const {
            return reverse_iterator{ ptr - x };
        }
    };
public:
    typedef T value_type;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef size_t size_type;
    enum { static_size = N };
    typedef std::ptrdiff_t difference_type;

    BOOST_MATH_CUDA_ENABLED std_array() = default;
    
    BOOST_MATH_CUDA_ENABLED std_array(std::initializer_list<T> init)
    {
        size_t i = 0;

        for (auto it = init.begin(); it != init.end() && i < N; ++it, ++i)
            storage_[i] = *it;

        for (; i < N; ++i)
            storage_[i] = T(0);
    }

    BOOST_MATH_CUDA_ENABLED reference operator[](size_type i)
    {
        return storage_[i];
    }
    BOOST_MATH_CUDA_ENABLED const_reference operator[](size_type i) const
    {
        return storage_[i];
    }
    BOOST_MATH_CUDA_ENABLED forward_iterator begin (void) {
        return forward_iterator{storage_};
    }
    BOOST_MATH_CUDA_ENABLED forward_iterator end (void) {
        return forward_iterator{storage_ + N};
    }

    BOOST_MATH_CUDA_ENABLED size_t size (void) const {
        return N;
    }
    BOOST_MATH_CUDA_ENABLED forward_iterator begin (void) const {
        return forward_iterator{storage_};
    }
    BOOST_MATH_CUDA_ENABLED forward_iterator end (void) const {
        return forward_iterator{storage_ + N};
    }

    BOOST_MATH_CUDA_ENABLED T const* cbegin (void) const {
        return &storage_[0];
    }
    BOOST_MATH_CUDA_ENABLED T const* cend (void) const {
        return &storage_[N];
    }

    BOOST_MATH_CUDA_ENABLED T& front (void) {
        return storage_[0];
    }
    BOOST_MATH_CUDA_ENABLED T const& front (void) const {
        return storage_[0];
    }
    BOOST_MATH_CUDA_ENABLED T& back (void) {
        return storage_[N - 1];
    }
    BOOST_MATH_CUDA_ENABLED T const& back (void) const {
        return storage_[N - 1];
    }
    BOOST_MATH_CUDA_ENABLED reverse_iterator rbegin (void) {
        return reverse_iterator{&storage_[N - 1]};
    }
    BOOST_MATH_CUDA_ENABLED reverse_iterator rbegin (void) const {
        return reverse_iterator{&storage_[N - 1]};
    }
    BOOST_MATH_CUDA_ENABLED reverse_iterator rend (void) {
        return reverse_iterator{&storage_[-1]};
    }
    BOOST_MATH_CUDA_ENABLED reverse_iterator rend (void) const {
        return reverse_iterator{&storage_[-1]};
    }
    BOOST_MATH_CUDA_ENABLED reverse_iterator crbegin (void) const {
        return reverse_iterator{&storage_[N - 1]};
    }
    BOOST_MATH_CUDA_ENABLED reverse_iterator crend (void) const {
        return reverse_iterator{&storage_[-1]};
    }
    BOOST_MATH_CUDA_ENABLED T& at (size_t i) {
        assert(i < N);
        return storage_[i];
    }
    BOOST_MATH_CUDA_ENABLED T const& at (size_t i) const {
        assert(i < N);
        return storage_[i];
    }

private:
    T storage_[N];
};

#endif /* __CUDACC__ */

} // namespace differentiation
} // namespace math
} // namespace boost

#endif /* ! BOOST_MATH_DIFFERENTIATION_ARRAY_HPP */
