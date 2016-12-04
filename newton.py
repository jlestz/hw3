# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import functions as F

class Newton(object):
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, Df=None):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian
        r: maximum search radius (exception thrown if |x_k - x0|>r)
        Df: optional argument specifying analytic Jacobian"""
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self._Df = Df


    def solve(self, x0, r=float("inf")):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
        for i in xrange(self._maxiter+1):
            fx = self._f(x)
            # if |x - x0| > r, throw an exception 
            if N.linalg.norm(x - x0) > r: 
                raise RuntimeError("Distance x - x0 exceeds search radius r")
            if N.linalg.norm(fx) < self._tol:
                return x
            # if i = maxiter, then maxiter steps have already been taken
            # and no solutions have been found. Raise an error. 
            elif i == self._maxiter: 
                raise RuntimeError("No solution found after maxiter iterations")
            x = self.step(x, fx)

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        
        Df = self._Df
        if Df is None: 
            # use numerical Jacobian 
            Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        else: 
            # analytic Jacobian option 
            Df_x = Df(x) 

        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        return x - h
