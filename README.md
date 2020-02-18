# Fitting models with uncertainty in dependent and independent varaibles

## Usage

To run a simple test, run the `labfit.py` file.

To fit using your own model and data, you have two options:
- put the file (`labfit.py`) in the same directory as your code and add `from labfit import fit` to your code
- put the imports and the top three functions in `labfit.py` (`_sq_md()`, `_residuals()`, and `fit()`) directly in your code

I might make this pip-installable at some point, but not yet.

With the proper code imported/copy-pasted, call `fit()` the same way you call
`scipy.optimize.curve_fit()`. The only difference is that you can specify an
`xerr` parameter. Note that you don't need to include `xerr` -- if you don't,
`fit()` will simply return the same result as `curve_fit()`. However, if you
do include `xerr`, you MUST include nonzero `yerr`.

## Algorithm

The basic idea is to minimize the sum of the squared Mahalanobis distances between each
data point and the model. In principle, if we had a simple function to compute
the Mahalanobis distance between a point and a curve, this would be
straightforward -- simply pass this function to `scipy.optimize.leastsq()` or
some other optimization code to find the model parameters that minimize this
distance.

However, we don't have a direct way to calculate the points on the curve to use
when calculating these distances. There are special cases, such as linear models,
where there is an analytic solution for this, but since the goal is to fit any
model, we can't use this.

Instead, each time the Mahalanobis distance function is called, we have a
separate subproblem: find the points on the curve that, given the current
model parameters, minimize the total squared Mahalanobis distance -- this is
done with a call to `scipy.optimize.minimize()`.

Because each step of the outer minimization requires solving its own (sometimes
high-dimensional) optimization problem, the code first uses
`scipy.optimize.curve_fit` to fit the model without `xerr`. It then uses the
result of this fit as the starting point for the full fit, minimizing the
number of Mahalanobis distance calculations that are required.
