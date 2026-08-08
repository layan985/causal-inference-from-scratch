import numpy as np
from causal_scratch.rdd import local_linear_rdd
from causal_scratch.synthetic_control import synthetic_control
from causal_scratch.matching import logistic_propensity, nearest_neighbor_match
from causal_scratch.rct import difference_in_means, randomization_inference
from causal_scratch.hte import linear_cate


def test_rdd_jump():
    rng=np.random.default_rng(5)
    x=rng.uniform(-2,2,12000)
    y=.5*x+2*(x>=0)+rng.normal(scale=.15,size=len(x))
    r=local_linear_rdd(y,x,bandwidth=.8)
    assert abs(r.tau-2) < .08


def test_synthetic_control_weights_simplex_and_fit():
    t=np.arange(8.0)
    donors=np.column_stack([t, 2*t+1, np.sin(t)])
    treated=.7*donors[:,0]+.3*donors[:,1]
    r=synthetic_control(treated,donors)
    assert abs(r.weights.sum()-1)<1e-9
    assert np.all(r.weights>=-1e-12)
    assert r.pre_rmse < .02


def test_matching_effect():
    rng=np.random.default_rng(6); n=1500
    x=rng.normal(size=(n,2)); logits=.7*x[:,0]-.4*x[:,1]
    p=1/(1+np.exp(-logits)); t=rng.binomial(1,p)
    y=1.8*t + x[:,0] + rng.normal(scale=.2,size=n)
    phat,_=logistic_propensity(t,x)
    m=nearest_neighbor_match(y,t,phat,caliper=.05)
    assert abs(m.att-1.8) < .18


def test_rct_and_randomization_inference():
    rng=np.random.default_rng(7); n=400
    t=np.array([0]*(n//2)+[1]*(n//2))
    y=rng.normal(size=n)+1.2*t
    r=difference_in_means(y,t)
    assert abs(r.ate-1.2)<.25
    ri=randomization_inference(y,t,reps=999,seed=7)
    assert ri['p_value_two_sided'] < .01


def test_linear_cate():
    rng=np.random.default_rng(8); n=6000
    x=rng.normal(size=(n,1)); t=rng.binomial(1,.5,size=n)
    tau=1+2*x[:,0]
    y=.5*x[:,0]+tau*t+rng.normal(scale=.2,size=n)
    r=linear_cate(y,t,x)
    assert abs(r.baseline-1)<.06
    assert abs(r.modifiers[0]-2)<.06
