## Locating Instanton

This script generalised the Cerjan&ndash;Miller algorithm slightly so that it is able to locate saddle points on a potential energy surface of given index (under suitable initial conditions). This makes it useful to locate the instanton configurations over a potential barrier, since an instanton making $n$ cycles in its imaginary-time trajectory over the inverted barrier is an order $2n-1$ saddle on the extended phase space of the ring polymer.

The original Cerjan&ndash;Miller algorithm chooses to locate the Lagrange multiplier between the lowest two eigenvalues of the Hessian matrix, so it is aiming at an index-1 saddle (transition state). Here we simply change it to find $\lambda$ as the local minimum of $\Delta^2(\lambda)$ between the $(k-1)$-th and $k$-th Hessian eigenvalues so it is aiming at a $k$-th order saddle. This is still far from perfect, though. For example, if you are aiming at an index 3 saddle, but the starting point is nearer to an index-5 one, then it will get attracted to the higher index one instead. It has no problem skipping the lower index saddle nonetheless.
 
This is illustrated with an asymmetric Eckart barrier in this script.

## References
 - Cerjan, C. J., & Miller, W. H. (1981). On finding transition states. The Journal of chemical physics, 75(6), 2800-2806.
