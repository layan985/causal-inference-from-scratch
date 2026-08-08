import numpy as np
from causal_scratch.linear import ols, cluster_robust_covariance
from causal_scratch.panel import within_estimator, two_way_demean


def test_ols_recovers_slope():
    x = np.arange(50.0)
    y = 3 + 2*x
    r = ols(y, x)
    assert np.allclose(r.beta, [3,2], atol=1e-10)


def test_within_removes_entity_intercepts():
    rng = np.random.default_rng(2)
    n_i, T = 80, 5
    entity = np.repeat(np.arange(n_i), T)
    x = rng.normal(size=n_i*T)
    a = np.repeat(rng.normal(scale=4,size=n_i), T)
    y = a + 1.7*x + rng.normal(scale=.05,size=n_i*T)
    r = within_estimator(y, x, entity)
    assert abs(r.beta[0]-1.7) < .03


def test_two_way_demean_zero_group_means():
    entity = np.repeat(np.arange(10), 4)
    time = np.tile(np.arange(4), 10)
    v = 2*entity + 3*time + np.sin(entity)
    d = two_way_demean(v, entity, time)
    for g in np.unique(entity): assert abs(d[entity==g].mean()) < 1e-9
    for tt in np.unique(time): assert abs(d[time==tt].mean()) < 1e-9


def test_cluster_covariance_shape():
    x = np.column_stack([np.ones(20), np.arange(20)])
    e = np.linspace(-1,1,20)
    c = np.repeat(np.arange(5),4)
    V = cluster_robust_covariance(x,e,c)
    assert V.shape == (2,2)
    assert np.all(np.diag(V) >= 0)
