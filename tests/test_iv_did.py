import numpy as np
import pandas as pd
from causal_scratch.iv import two_stage_least_squares
from causal_scratch.did import did_2x2, group_time_att, aggregate_event_time


def test_iv_recovers_effect():
    rng=np.random.default_rng(3); n=6000
    z=rng.normal(size=n); u=rng.normal(size=n)
    x=.9*z+u+rng.normal(scale=.4,size=n)
    y=2.5*x+u+rng.normal(scale=.4,size=n)
    r=two_stage_least_squares(y,x,z)
    assert abs(r.second_stage.beta[1]-2.5) < .08
    assert r.first_stage_f > 10


def test_did_2x2():
    y=[1,2,2,3,1,2,3,4]
    treated=[0,0,0,0,1,1,1,1]
    post=[0,0,1,1,0,0,1,1]
    assert abs(did_2x2(y,treated,post).att-1.0) < 1e-12


def test_group_time_att_recovers_dynamic_effect():
    rows=[]
    rng=np.random.default_rng(4)
    for i in range(200):
        g=3 if i<70 else (4 if i<140 else np.nan)
        for t in range(1,7):
            eff=0 if np.isnan(g) or t<g else (t-g+1)*1.5
            rows.append((i,t,g, i*.01 + .2*t + eff + rng.normal(scale=.03)))
    df=pd.DataFrame(rows,columns=['id','t','g','y'])
    gt=group_time_att(df,'y','id','t','g')
    agg=aggregate_event_time(gt)
    first=float(agg.loc[agg.event_time==0,'att'].iloc[0])
    assert abs(first-1.5) < .12
