#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    # tests Jacobian for a 1D linear function : R^1 --> R^1 
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    # tests Jacobian for a constant function : R^2 --> R^2
    # importantly, this verifies that the Jacobian isn't transposed 
    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    # tests Jacobian for a constant function : R^2 --> R^1 
    # (non square Jacobian test)
    def testApproxJacobian3(self): 
        A = N.matrix("1. 2.") 
        def f(x): 
            return A * x
        x0 = N.matrix("5 ; 6")
        dx=1.e-6
        Df_x = F.ApproximateJacobian(f,x0,dx)
        self.assertEqual(Df_x.shape, (1,2))
        N.testing.assert_array_almost_equal(Df_x,A)

    # tests Jacobian for a constant function : R^1 --> R^2
    # (non square Jacobian test)
    def testApproxJacobian4(self): 
        A = N.matrix("1. ; 2.") 
        def f(x): 
            return A * x
        x0 = 2.0
        dx=1.e-6
        Df_x = F.ApproximateJacobian(f,x0,dx)
        self.assertEqual(Df_x.shape, (2,1))
        N.testing.assert_array_almost_equal(Df_x,A)

    # tests the accuracy of polynomial function 
    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    # test Jacobian against analytically known Jacobian of elementary functions 
    def testTrig(self): 
        def f(x): 
            return N.sin(x[0])*N.cos(x[1])
        x0 = N.matrix("2.0 ; 1.0")
        dx=1.e-6
        Df_x = F.ApproximateJacobian(f,x0,dx)
        Df_anal = N.matrix([N.cos(2.0)*N.cos(1.0) ,-1.0*N.sin(2.0)*N.sin(x0[1.0])])
        self.assertEqual(Df_x.shape, (1,2))
        N.testing.assert_array_almost_equal(Df_x,Df_anal); 

if __name__ == '__main__':
    unittest.main()



