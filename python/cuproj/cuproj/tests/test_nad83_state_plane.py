# Copyright (c) 2024, NVIDIA CORPORATION.

import cupy as cp
import geopandas as gpd
import numpy as np
import pytest
from cupy.testing import assert_allclose
from pyproj import Transformer
from shapely.geometry import Point

import cuspatial
from cuproj import Transformer as cuTransformer


def test_nad83_state_plane_single_point():
    # Test a single point in San Francisco
    lat = 37.7749
    lon = -122.4194

    # Transform using PyProj for comparison
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2227")
    pyproj_x, pyproj_y = transformer.transform(lat, lon)

    # Transform using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "EPSG:2227")
    cuproj_x, cuproj_y = cu_transformer.transform(lat, lon)

    # Allow for small differences in the transformation
    assert_allclose(cuproj_x, pyproj_x, rtol=1e-6)
    assert_allclose(cuproj_y, pyproj_y, rtol=1e-6)


def test_nad83_state_plane_multiple_points():
    # Test multiple points around California
    points = [
        (37.7749, -122.4194),  # San Francisco
        (34.0522, -118.2437),  # Los Angeles
        (38.5816, -121.4944),  # Sacramento
        (32.7157, -117.1611),  # San Diego
    ]

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    # Transform using PyProj for comparison
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2227")
    pyproj_x, pyproj_y = transformer.transform(lats, lons)

    # Transform using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "EPSG:2227")
    cuproj_x, cuproj_y = cu_transformer.transform(lats, lons)

    # Allow for small differences in the transformation
    assert_allclose(cuproj_x, pyproj_x, rtol=1e-6)
    assert_allclose(cuproj_y, pyproj_y, rtol=1e-6)


def test_nad83_state_plane_geoseries():
    # Test with cuspatial GeoSeries input
    points = [
        Point(-122.4194, 37.7749),  # San Francisco
        Point(-118.2437, 34.0522),  # Los Angeles
    ]

    s = gpd.GeoSeries(points)
    gs = cuspatial.from_geopandas(s)

    # Transform using PyProj for comparison
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2227")
    pyproj_x, pyproj_y = transformer.transform(s.y.values, s.x.values)

    # Transform using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "EPSG:2227")
    cuproj_x, cuproj_y = cu_transformer.transform(gs.points.y, gs.points.x)

    # Allow for small differences in the transformation
    assert_allclose(cuproj_x, pyproj_x, rtol=1e-6)
    assert_allclose(cuproj_y, pyproj_y, rtol=1e-6)


def test_nad83_state_plane_inverse():
    # Test inverse transformation from State Plane back to WGS84
    # Using known State Plane coordinates for San Francisco
    x = 2000000.0  # Approximate State Plane X coordinate
    y = 500000.0   # Approximate State Plane Y coordinate

    # Transform using PyProj for comparison
    transformer = Transformer.from_crs("EPSG:2227", "EPSG:4326")
    pyproj_lat, pyproj_lon = transformer.transform(x, y)

    # Transform using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:2227", "EPSG:4326")
    cuproj_lat, cuproj_lon = cu_transformer.transform(x, y)

    # Allow for small differences in the transformation
    assert_allclose(cuproj_lat, pyproj_lat, rtol=1e-6)
    assert_allclose(cuproj_lon, pyproj_lon, rtol=1e-6)


def test_nad83_state_plane_different_precisions():
    # Test with different numeric precisions (float32 and float64)
    lat = np.array([37.7749], dtype=np.float32)
    lon = np.array([-122.4194], dtype=np.float32)

    # Transform using PyProj for comparison
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2227")
    pyproj_x, pyproj_y = transformer.transform(lat, lon)

    # Transform using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "EPSG:2227")
    cuproj_x, cuproj_y = cu_transformer.transform(lat, lon)

    # Allow for larger differences with float32
    assert_allclose(cuproj_x, pyproj_x, rtol=1e-4)
    assert_allclose(cuproj_y, pyproj_y, rtol=1e-4) 