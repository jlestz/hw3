#!/usr/bin/env python

import newton
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    # test if correct root is found for linear function : R^1 --> R^1
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

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
        self.assertEqual(x1,1.0)
    
if __name__ == "__main__":
    unittest.main()
