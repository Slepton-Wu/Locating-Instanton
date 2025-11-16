## Locating Instanton

This script generalised the Cerjan&ndash;Miller algorithm slightly so that it is able to locate saddle points on a potential energy surface of given index (under suitable initial conditions). This makes it useful to locate the instanton configurations over a potential barrier, since an instanton making $n$ cycles in its imaginary-time trajectory over the inverted barrier is an order $2n-1$ saddle on the extended phase space of the ring polymer.

The original Cerjan&ndash;Miller algorithm chooses to locate the Lagrange multiplier between the lowest two eigenvalues of the Hessian matrix, so it is aiming at an index-1 saddle (transition state). Here we simply change it to find $\lambda$ as the local minimum of $\Delta^2(\lambda)$ between the $(k-1)$-th and $k$-th Hessian eigenvalues so it is aiming at a $k$-th order saddle. 

There are two versions of the script, one with 'strict' and another without. The one with 'strict' always uses Cerjan&ndash;Miller algorithm, unless the order of Hessian there is the same as the order of the saddle you are aiming at, in which case it switches to Newton&ndash;Raphson. The non-strict one uses Cerjan&ndash;Miller only if the order of the Hessian is smaller than the order of the saddle you set to aim at, otherwise it uses Newton&ndash;Raphson. The non-strict one will be attracted by higher order saddles, i.e. if you are aiming at a index-3 saddle, but you move near to an index-5 saddle, then it will get attracted by this higher order saddle and terminate there. The strict version does not, but it takes a larger number of steps to converge.

If you don't care the order of the saddle point you find, use the non-strict version with the targeting saddle order set as 1, then it will find its closest saddle of arbitrary order. If you really want a saddle of specified order, use the strict version.

This is illustrated with an asymmetric Eckart barrier in this script.

## References

- Cerjan, C. J., & Miller, W. H. (1981). On finding transition states. The Journal of chemical physics, 75(6), 2800-2806.
