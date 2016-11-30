#!/usr/bin/env python

import newton
import unittest
import numpy as N
from scipy import special

class TestNewton(unittest.TestCase):
    # test if correct root is found for linear function : R^1 --> R^1
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    # test if correct root is found for trig functions  
    def testTrig(self):
        f = lambda x : N.sin(x)
        g = lambda x : N.cos(x)
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.solve(1.0)
        solver = newton.Newton(g, tol=1.e-15,maxiter=10) 
        y=solver.solve(1.0)
        self.assertAlmostEqual(x,0.0)
        self.assertAlmostEqual(y,N.pi/2)

    # test if correct root is found for Airy function 
    def testAiry(self):
        def fAiry(x):
            f = special.airy(x)
            return f[0]
        f = fAiry
        solver=newton.Newton(f, tol=1.e-15,maxiter=100)
        x = solver.solve(-1.5)
        x0 = special.ai_zeros(1)
        x0 = x0[0]
        self.assertAlmostEqual(x,x0)

    # test if exception is thrown for function with no root 
    def testNoRoot(self): 
        f = lambda x : x**2 + 1.0 
        solver = newton.Newton(f, tol=1.e-15, maxiter=100) 
        try:  
            x = solver.solve(1.0)
        except RuntimeError: 
            pass 
        else: 
            self.fail('No error raised'); 
        
    # test that the correct step is taken in a known case 
    def testStep(self): 
        f = lambda x : x + 1.0
        solver = newton.Newton(f, tol=1.e-15,maxiter=2)
        x1 = solver.step(2.0)
        self.assertAlmostEqual(x1,-1.0)
    
    # verify that r = 0 always halts Newton Solver
    def testRadiusZero(self): 
        f = lambda x : x + 1.0
        solver = newton.Newton(f,r=0.0)
        try: 
            x = solver.solve(1.0) 
        except RuntimeError: 
            pass 
        else: 
            self.fail('No error raised'); 

    # test for a constant function (no root, singular Jacobian)
    # should throw exception since singular Jacobian breaks method
    def testConstant(self): 
        f = lambda x: 1.0
        solver = newton.Newton(f)
        try: 
            x = solver.solve(1.0) 
        except N.linalg.LinAlgError: 
            pass 
        else: 
            self.fail('No error raised'); 

    # verify 

if __name__ == "__main__":
    unittest.main()
