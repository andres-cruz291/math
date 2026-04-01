
//  Copyright Andres Cruz 2026.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff_cuda_1.cu"
#include "test_autodiff_cuda_2.cu"
#include "test_autodiff_cuda_3.cu"

/**
 * Host main routine
 */
int main(void)
{
    if (!main_tests_1<double>(32))
        return EXIT_FAILURE;
    if (!main_tests_2<double>(32))
        return EXIT_FAILURE;
    if (!main_tests_3<double>(32))
        return EXIT_FAILURE;

    return 0;
}
