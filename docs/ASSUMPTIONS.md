# Identification assumptions: say them before you estimate

| Design | Target | Core assumptions | Fastest way to embarrass yourself |
|---|---|---|---|
| OLS causal regression | conditional average effect | exogeneity conditional on controls; correct target population; no post-treatment controls | calling a coefficient causal because it has fixed effects |
| Fixed effects | within-unit effect | time-varying confounding absent after FE; enough within variation | assuming FE solve all omitted-variable bias |
| IV / 2SLS | LATE | relevance, exclusion, independence, monotonicity | celebrating a significant first stage while exclusion is nonsense |
| RDD | local treatment effect at cutoff | continuity of potential outcomes; no precise manipulation; correct bandwidth/specification | choosing bandwidth after seeing the preferred p-value |
| 2x2 DiD | ATT | parallel trends in untreated potential outcomes | treating pre-trend non-rejection as proof |
| Staggered DiD | dynamic ATT | valid cohort-specific comparisons; no anticipation | reporting vanilla TWFE with heterogeneous effects as the answer |
| Synthetic control | treated-unit counterfactual | donor pool can approximate untreated path; no spillovers | using donors exposed to the intervention |
| Matching | ATT/ATE under selection on observables | conditional exchangeability, positivity, correct overlap | claiming matching fixes unobserved confounding |
| RCT | ATE | random assignment, SUTVA/appropriate interference model, attrition handled | conditioning on post-randomization variables |
| HTE | CATE | baseline identification assumptions plus honest multiplicity/model discipline | data-mining subgroups until something looks exciting |
