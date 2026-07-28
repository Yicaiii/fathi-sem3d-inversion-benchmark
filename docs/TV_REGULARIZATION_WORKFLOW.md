# TV regularization extension

The verified data-only benchmark is preserved as the baseline.

The TV extension inserts the regularization contribution before the
Mtilde solve:

data RHS + weighted TV RHS -> Mtilde solve -> candidates

For the homogeneous lambda = mu = 80 MPa initial model, the spatial
TV gradient should be zero up to numerical precision. Consequently,
the first transition should agree with the data-only direction.

Once the accepted model becomes spatially non-uniform, TV weights
must be recomputed from the current transition. Alpha values from
iter_008 must not be reused for iter_000 or later unrelated states.

The generic iteration sequence is:

1. create the iteration context;
2. prepare strict forward;
3. run strict forward;
4. generate residual sources;
5. prepare and run adjoints;
6. assemble data RHS;
7. compute and restrict TV RHS;
8. solve Mtilde;
9. generate candidates;
10. run candidate forward evaluations;
11. accept only a descending regularized objective.
