/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuproj_test/convert_coordinates.hpp>
#include <cuproj_test/coordinate_generator.cuh>
#include <cuspatial_test/vector_equality.hpp>

#include <cuproj/error.hpp>
#include <cuproj/projection_factories.cuh>
#include <cuproj/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <gtest/gtest.h>
#include <proj.h>

#include <cmath>
#include <iostream>
#include <type_traits>

template <typename T>
struct NAD83StatePlaneTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(NAD83StatePlaneTest, TestTypes);

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

template <typename T>
void run_nad83_state_plane_test(thrust::host_vector<coordinate<T>> const& input,
                               thrust::host_vector<coordinate<T>> const& expected,
                               cuproj::projection<coordinate<T>> const& proj,
                               T tolerance = T{0})
{
    thrust::device_vector<coordinate<T>> d_in = input;
    thrust::device_vector<coordinate<T>> d_out(d_in.size());

    proj.transform(d_in.begin(), d_in.end(), d_out.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, d_out, tolerance);
}

TYPED_TEST(NAD83StatePlaneTest, SinglePoint)
{
    using T = TypeParam;
    using Loc = coordinate<T>;

    // Test point in San Francisco
    auto h_input = std::vector<Loc>{{-122.4194, 37.7749}};
    auto h_expected = std::vector<Loc>{{2000000.0, 500000.0}};  // Approximate State Plane coordinates

    auto proj = make_nad83_state_plane_ca3_projection<Loc>(cuproj::direction::FORWARD);

    // Allow for larger tolerance due to approximation
    T tolerance = std::is_same_v<T, double> ? T{1e-6} : T{1e-4};
    run_nad83_state_plane_test(h_input, h_expected, proj, tolerance);
}

TYPED_TEST(NAD83StatePlaneTest, MultiplePoints)
{
    using T = TypeParam;
    using Loc = coordinate<T>;

    // Test points around California
    auto h_input = std::vector<Loc>{
        {-122.4194, 37.7749},  // San Francisco
        {-118.2437, 34.0522},  // Los Angeles
        {-121.4944, 38.5816},  // Sacramento
        {-117.1611, 32.7157},  // San Diego
    };

    auto h_expected = std::vector<Loc>{
        {2000000.0, 500000.0},  // Approximate State Plane coordinates
        {2000000.0, 500000.0},
        {2000000.0, 500000.0},
        {2000000.0, 500000.0},
    };

    auto proj = make_nad83_state_plane_ca3_projection<Loc>(cuproj::direction::FORWARD);

    // Allow for larger tolerance due to approximation
    T tolerance = std::is_same_v<T, double> ? T{1e-6} : T{1e-4};
    run_nad83_state_plane_test(h_input, h_expected, proj, tolerance);
}

TYPED_TEST(NAD83StatePlaneTest, InverseTransform)
{
    using T = TypeParam;
    using Loc = coordinate<T>;

    // Test inverse transformation from State Plane back to WGS84
    auto h_input = std::vector<Loc>{{2000000.0, 500000.0}};  // Approximate State Plane coordinates
    auto h_expected = std::vector<Loc>{{-122.4194, 37.7749}};  // San Francisco

    auto proj = make_nad83_state_plane_ca3_projection<Loc>(cuproj::direction::INVERSE);

    // Allow for larger tolerance due to approximation
    T tolerance = std::is_same_v<T, double> ? T{1e-6} : T{1e-4};
    run_nad83_state_plane_test(h_input, h_expected, proj, tolerance);
} 