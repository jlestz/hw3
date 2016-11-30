# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import functions as F

class Newton(object):
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, r=float("inf")):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian
        r: maximum search radius (exception thrown if |x_k - x0|>r)"""
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self._r = r

    def solve(self, x0):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
        r = self._r
        for i in xrange(self._maxiter+1):
            # if |x - x0| > r, throw an exception 
            if N.linalg.norm(x - x0) > r: 
                raise RuntimeError("Distance x - x0 exceeds search radius r")
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            # if i = maxiter, then maxiter steps have already been taken
            # and no solutions have been found. Raise an error. 
            elif i == self._maxiter: 
                raise RuntimeError("No solution found after max_iter iterations")
            x = self.step(x, fx)

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        return x - h
