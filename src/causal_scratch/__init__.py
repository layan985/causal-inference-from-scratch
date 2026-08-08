"""Small, transparent causal-inference implementations for learning and testing."""

from .linear import ols, cluster_robust_covariance
from .panel import within_estimator, two_way_demean
from .iv import two_stage_least_squares
from .did import did_2x2, group_time_att, aggregate_event_time
from .rdd import local_linear_rdd
from .synthetic_control import synthetic_control
from .matching import logistic_propensity, nearest_neighbor_match
from .rct import difference_in_means, randomization_inference
from .hte import linear_cate

__all__ = [
    "ols", "cluster_robust_covariance", "within_estimator", "two_way_demean",
    "two_stage_least_squares", "did_2x2", "group_time_att", "aggregate_event_time",
    "local_linear_rdd", "synthetic_control", "logistic_propensity", "nearest_neighbor_match",
    "difference_in_means", "randomization_inference", "linear_cate",
]
