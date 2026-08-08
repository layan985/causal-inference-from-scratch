import numpy as np
from causal_scratch import did_2x2, local_linear_rdd, two_stage_least_squares

rng=np.random.default_rng(42)
print("2x2 DiD:", did_2x2([1,1,2,2,1,1,3,3],[0,0,0,0,1,1,1,1],[0,0,1,1,0,0,1,1]).att)

x=rng.uniform(-1,1,5000)
y=x+2*(x>=0)+rng.normal(scale=.2,size=len(x))
print("RDD jump:", round(local_linear_rdd(y,x,bandwidth=.5).tau,3))

z=rng.normal(size=3000); u=rng.normal(size=3000); x=.8*z+u+rng.normal(size=3000); y=3*x+u+rng.normal(size=3000)
print("IV slope:", round(two_stage_least_squares(y,x,z).second_stage.beta[1],3))
