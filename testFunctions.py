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
# currently fails because shape is not equal...
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
# currently has some error -- need to read up on numpy
    def testApproxJacobian4(self): 
        A = N.matrix("1. ; 2.") 
        def f(x): 
            return A * x
        x0 = 2.0
        dx=1.e-6
        Df_x = F.ApproximateJacobian(f,x0,dx)
        self.assertEqual(Df_x.shape, (2,1))
        N.testing.assert_array_almost_equal(Df_x,A)

    # tests Jacobian for a 2nd degree polynomial in 1D : R^1 --> R^1 
    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

# test Jacobian against analytically known Jacobian 

if __name__ == '__main__':
    unittest.main()



