import pytest
import numpy as np
import astropy.time

import egelib.timeseries

@pytest.mark.parametrize("t_fmt,dur_fmt", [
  ('jyear', 'yr'),
  ('jd', 'd'),
  ('unix', 's')
])
def test_duration(t_fmt, dur_fmt):
    start = 2000.0
    dur = 10.0
    tstart = astropy.time.Time(start, format='jyear', scale='utc')
    tstart = getattr(tstart, t_fmt)
    N = 10
    t = np.linspace(tstart, tstart + dur, N)
    t = astropy.time.Time(t, format=t_fmt, scale='utc')

    d = egelib.timeseries.duration(t, dur_fmt)
    np.testing.assert_allclose(d, dur)

@pytest.mark.parametrize("t_fmt,dur_fmt", [
  ('jyear', 'yr'),
  ('jd', 'd'),
  ('unix', 's')
])
def test_intervals(t_fmt, dur_fmt):
    start = 2000.0
    dur = 10.0
    tstart = astropy.time.Time(start, format='jyear', scale='utc')
    tstart = getattr(tstart, t_fmt)
    N = 10
    t = np.linspace(tstart, tstart + dur, N+1)
    t = astropy.time.Time(t, format=t_fmt, scale='utc')

    d = egelib.timeseries.intervals(t, dur_fmt)
    np.testing.assert_allclose(d, dur/N, rtol=1e-6) # microsecond precision in 's' case

def test_find_coinc():
    t1 = astropy.time.Time([2000.0, 2012.5, 2010.0], format='jyear', scale='utc')
    t2 = astropy.time.Time([2012.6, 2015.0, 2020.0], format='jyear', scale='utc')
    t1c, t2c  = egelib.timeseries.find_coincident(t1, t2, 1.0, 'yr')
    assert len(t1c) == 1
    assert len(t2c) == 1
    assert t1c[0] == 1
    assert t2c[0] == 0

@pytest.mark.parametrize("t1,t2,expect", [
    (np.linspace(2000.0, 2010.0, 10), np.linspace(2005.0, 2007.0, 10), (2005.0, 2007.0)), # t1 contains t2
    (np.linspace(2000.0, 2010.0, 10), np.linspace(2005.0, 2020.0, 10), (2005.0, 2010.0)), # overlap right
    (np.linspace(2000.0, 2010.0, 10), np.linspace(1995.0, 2005.0, 10), (2000.0, 2005.0)), # overlap left
    (np.linspace(2000.0, 2010.0, 10), np.linspace(1995.0, 2020.0, 10), (2000.0, 2010.0)), # t2 contains t1
    (np.linspace(2000.0, 2010.0, 10), np.linspace(1995.0, 1996.0, 10), None),             # no overlap
])
def test_overlap(t1, t2, expect):
    t1 = astropy.time.Time(t1, format='jyear', scale='utc')
    t2 = astropy.time.Time(t2, format='jyear', scale='utc')
    overlap = egelib.timeseries.overlap(t1, t2)
    if expect is not None:
        assert overlap[0].jyear == expect[0] and overlap[1].jyear == expect[1]
    else:
        assert overlap[0] is None and overlap[1] is None

def test_overlap_calib():
    N1 = 10
    mean1 = 2.3
    t1 = np.linspace(2000.0, 2010.0, N1)
    y1 = np.ones(N1) * mean1

    N2 = 13
    mean2 = 4.5
    t2 = np.linspace(2005.0, 2007.0, N2)
    y2 = np.ones(N2) * mean2

    C = egelib.timeseries.overlap_calib(t1, y1, t2, y2)
    np.testing.assert_allclose(C, mean1/mean2)

def test_season_offset():
    t = np.arange(2000.0, 2010.0, 1.0) # one measure every year
    t = astropy.time.Time(t, format='jyear', scale='utc')
    offset = egelib.timeseries.season_offset(t)
    np.testing.assert_allclose(offset, 0.5) # offset is in middle of year

def test_season_edges():
    t = np.arange(2000.0, 2010.0, 1.0) # one measure every year
    t_ast = astropy.time.Time(t, format='jyear', scale='utc')
    edges = egelib.timeseries.season_edges(t_ast) # edges containing every point
    offset = 0.5
    expect_edges = t - offset
    expect_edges = np.append(expect_edges, 2009 + offset)
    np.testing.assert_allclose(edges.value, expect_edges)

def test_season_indices():
    t = [2005.4, 2005.5, 2005.6, 2006.4, 2006.5, 2006.6]
    t_ast = astropy.time.Time(t, format='jyear', scale='utc')
    seasons = egelib.timeseries.season_indices(t_ast)
    assert np.array_equal(seasons[0], [0, 1, 2])
    assert np.array_equal(seasons[1], [3, 4, 5])

def test_seasonal_series():
    t = [2005.4, 2005.5, 2005.6, 2006.4, 2006.5, 2006.6]
    t_ast = astropy.time.Time(t, format='jyear', scale='utc')
    y = np.ones_like(t)
    ts, ys = egelib.timeseries.seasonal_series(t_ast, y)
    assert np.array_equal(ts[0].value, t[0:3])
    assert np.array_equal(ts[1].value, t[3:])

def test_seasonal_means():
    t = [2005.4, 2005.5, 2005.6, 2006.4, 2006.5, 2006.6]
    t_ast = astropy.time.Time(t, format='jyear', scale='utc')
    y = np.ones_like(t)
    ts, ys, es, Ns = egelib.timeseries.seasonal_means(t_ast, y)
    np.testing.assert_allclose(ts.value, [2005.5, 2006.5])  # season midpoints
    np.testing.assert_allclose(ys, [1.0, 1.0])              # season means
    np.testing.assert_allclose(es, [0.0, 0.0])              # season standard devs
    np.testing.assert_array_equal(Ns, [3, 3])               # season counts
