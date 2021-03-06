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

    # test that analytic Jacobians are close to numerical
    # check that they match to at least 5 decimals when calculating Numerical Jacobian with dx = 1e-6 precision
    def checkAnalJacobian(self,fun,x,dx=1e-6):
        DfAnal = fun.Df(x)
        DfNum = F.ApproximateJacobian(fun,x,dx=dx)
        N.testing.assert_allclose(DfNum,DfAnal,rtol=10*dx,atol=10*dx)

    def testPolynomialJacobian(self):
        for x in range(-10,10): 
            self.checkAnalJacobian(F.Polynomial([3,0,4]),x)

    def testPolyLogJacobian(self): 
        for x in N.linspace(0.1,2,10):
            self.checkAnalJacobian(F.PolyLog([5,1,2],3),x)

    def testExpSinJacobian(self):
        testvals=N.linspace(-2*N.pi,2*N.pi,20); 
        for x in testvals:
            for y in testvals:
                self.checkAnalJacobian(F.ExpSin(2),[x,y],1.e-6)

    # test the PolyLog class for correct values  
    def testPolyLogVals(self):
        # check values when PolyLog reduces to Log
        p = F.PolyLog([1],1)
        self.assertEqual(p(1.0),0.0)
        self.assertEqual(p.Df(1.0),1.0)
        
        # check values when PolyLog reduces to Polynomial
        p = F.PolyLog([1,2,3],0)
        self.assertEqual(p(1.0),6.0)
        self.assertEqual(p.Df(1.5),5.0)

        # check values when PolyLog is a nontrivial product 
        p = F.PolyLog([1,-2,3],3)
        self.assertAlmostEqual(p(1.1),  0.00174026)
        self.assertAlmostEqual(p.Df(1.1),0.0499702)
        p = F.PolyLog([1, 0],1)
        x0=1.1
        self.assertEqual(p(x0),x0*N.log(x0))
        self.assertEqual(p.Df(x0),1 + N.log(x0))

    # test that ExpSin works for all all input types (list, array, matrix)
    def testExpSinInput(self): 
        # choose 3D input 
        p=F.ExpSin(3)
        
        # input types: list, array, matrix
        x=[0,-5,N.pi/2]; 
        xarr=N.array(x); 
        xmat=N.matrix(x); 
        
        # test equal values 
        self.assertEqual(p(x),p(xarr)); 
        self.assertEqual(p(xarr),p(xmat)); 

        # test equal analytic Jacobians 
        N.testing.assert_array_almost_equal(p.Df(x),p.Df(xarr))
        N.testing.assert_array_almost_equal(p.Df(xarr),p.Df(xmat))
        
        # test equal numerical Jacobians 
        N.testing.assert_array_almost_equal(F.ApproximateJacobian(p,x),F.ApproximateJacobian(p,xarr))
        N.testing.assert_array_almost_equal(F.ApproximateJacobian(p,xarr),F.ApproximateJacobian(p,xmat))

    # test the ExpSin class for correct values 
    def testExpSinVals(self): 
        # test 1D case 
        p = F.ExpSin(1)
        x0=0.5
        x1=0.2
        self.assertAlmostEqual(p(x0),N.exp(-x0**2)*N.sin(x0))
        self.assertAlmostEqual(p.Df(x1),N.exp(-x1**2)*(-2*x1*N.sin(x1) + N.cos(x1)))

        # test 6D case (arbitrary dimension) at x = zeros (possible edge case)
        dimen=6
        p = F.ExpSin(dimen)
        x0=N.zeros(dimen)
        self.assertEqual(p(x0),0.0)
        Df = p.Df(x0)
        sol = N.matrix(N.zeros(dimen))
        N.testing.assert_array_almost_equal(Df, sol)

if __name__ == '__main__':
    unittest.main()



