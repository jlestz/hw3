#!/usr/bin/env python

import newton
import functions as F 
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

    # test if maxiters exception is raised at correct step number 
    # (e.g. make sure loop is robust to edge cases of solutions that are found exactly 1 iteration after maxiters) 
    # with tol=2^-5, solution requires 3 iterations
    # maxiter = 2 should raise an error 
    def testMaxItersReached(self): 
        f = lambda x: x**2
        solver = newton.Newton(f,tol=2**-5,maxiter=2)
        x0=1.0 
        try: 
            x = solver.solve(x0) 
        except RuntimeError as err: 
            if "maxiter" in err.message: 
                pass 
            else: 
                self.fail('Wrong error raised')
        else: 
            self.fail('No error raised')
    
    # test if maxiters exception is raised at correct step number 
    # (e.g. make sure loop is robust to edge cases of solutions that are found exactly 1 iteration after maxiters) 
    # with tol=2^-5, solution requires 3 iterations 
    # maxiter = 3 should not raise an error 
    def testMaxItersNotReached(self): 
        f = lambda x: x**2
        solver = newton.Newton(f,tol=2**-5,maxiter=3)
        x0=1.0 
        try: 
            x = solver.solve(x0) 
        except RuntimeError as err: 
            if "maxiter" in err.message: 
                self.fail('Error raised erroneously')
        else: 
            self.assertAlmostEqual(x,2**-3,places=6)

    # test that the correct step is taken in a known case 
    def testStep(self): 
        f = lambda x : x + 1.0
        solver = newton.Newton(f, tol=1.e-15,maxiter=2)
        x1 = solver.step(2.0)
        self.assertAlmostEqual(x1,-1.0)
    
    # verify that r = 0 always halts Newton Solver
    def testRadiusZero(self): 
        f = lambda x : x + 1.0
        solver = newton.Newton(f)
        try: 
            x = solver.solve(1.0,r=0.0) 
        except RuntimeError: 
            pass 
        else: 
            self.fail('No error raised')

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
            self.fail('No error raised')

    # verify f(x) = e^x fails due to maxiters (within search radius)
    # for f(x) = e^x, x_{k+1} = x_k - 1
    # maxiters = 3 should fail before r = 4 for x0 = 0
    def testExpMaxIters(self): 
        f = lambda x: N.exp(x)
        solver = newton.Newton(f,tol=1.e-15,maxiter=3) 
        x0=0.0
        try: 
            x = solver.solve(x0,r=4) 
        except RuntimeError as err: 
            if "maxiter" in err.message: 
                pass 
            else: 
                self.fail('Wrong error raised')
        else: 
            self.fail('No error raised')

    # verify f(x) = e^x fails due to search radius (within maxiters)
    # for f(x) = e^x, x_{k+1} = x_k - 1
    # maxiters = 4 should not fail before r = 3 for x0 = 0
    def testExpMaxRadius(self): 
        f = lambda x: N.exp(x)
        solver = newton.Newton(f,tol=1.e-15,maxiter=4) 
        x0=0.0
        try: 
            x = solver.solve(x0,r=3) 
        except RuntimeError as err: 
            if "radius r" in err.message: 
                pass 
            else: 
                self.fail('Wrong error raised')
        else: 
            self.fail('No error raised')

    # test if radius exception is thrown when some intermediate steps leave the search radius even if the roots are known to be within search radius. f(x) = x^2 - 1 has roots at x = +/- 1, but an initial guess of x0=0.1 leads to a first step of x0 near 5, clearly beyond a radius of 2, despite the fact that subsequent steps would lead to convergence at x = 1
    def testStepBeyondSearch(self): 
        f = lambda x: x**2 - 1 
        solver = newton.Newton(f,maxiter=10) 
        x0=0.1 
        try: 
            x = solver.solve(x0,r=2) 
        except RuntimeError: 
            pass 
        else: 
            self.fail('No error raised') 

    # same test as above except r = 5 to accomodate the first step 
    # since the first step has the largest error, the solver will converge on the correct solution 
    def testStepBeyondSearchSuccess(self): 
        f = lambda x: x**2 - 1 
        solver = newton.Newton(f,maxiter=10) 
        x0=0.1 
        try: 
            x = solver.solve(x0,r=5) 
        except RuntimeError: 
            self.fail('Error raised erroneously')
        else: 
            self.assertAlmostEqual(x,1.0)

    # test analytic Jacobian: does it even run with new syntax
    def testAnalSin(self): 
        x0=1.0 
        solver = newton.Newton(N.sin,Df=N.cos)
        x = solver.solve(x0)
        self.assertAlmostEqual(x,0.0)

    # test if analytic Jacobian is used
    # 1. do the first steps differ? 
    # 2. do they arrive at the same solution? 
    def testPolyLogAnalVsNum(self): 
        # function (x-2)(x-4)log^2(x) has zeros at x=1,2,4
        x0=1.7
        p = F.PolyLog([1,-6,8],2); 

        dp = p.Df
        solverAnal = newton.Newton(p,tol=1.e-15,Df=dp)
        x1Anal=solverAnal.step(x0) 
        xSolAnal = solverAnal.solve(x0)

        solverNum = newton.Newton(p,tol=1.e-15,dx=1.e-8) 
        x1Num = solverNum.step(x0)
        xSolNum = solverNum.solve(x0) 

        self.assertNotEqual(x1Anal,x1Num) 
        self.assertEqual(xSolAnal,xSolNum)

    # tests solution for a 3D linear system : R^3 --> R^3
    def testSolution3D(self):
        A = N.matrix("8. 1. 6. ; 3. 5. 7. ; 4. 9. 2.")
        b = N.matrix("1. ; 2.; 3.")
        def f(x):
            x=N.matrix(x); 
            x=N.reshape(x,(N.size(x),1))
            ans = A * x - b
            return N.reshape(ans,(N.size(ans),1))
        x0 = N.matrix("0, 0, 0")
        y = N.matrix("0.05, 0.3, 0.05"); 
        solver=newton.Newton(f); 
        xsol=solver.solve(x0); 
        N.testing.assert_array_almost_equal(xsol, y)

    # tests solution for a 2D vector-valued function 
    def testSolution2D(self): 
        x0 = [1.1,2.2]
        f = F.Trig(); 
        solver=newton.Newton(f); 
        xsol=solver.solve(x0);
        x=xsol[0,0]; 
        y=xsol[0,1];
        s = x+y
        d = x-y
        stest = s/N.pi; 
        dtest = (d/N.pi) - 0.5; 
        # compare to analytic solutions for cos(x-y)=0,sin(x+y)=0
        self.assertAlmostEqual(stest,round(stest))
        self.assertAlmostEqual(dtest,round(dtest))

if __name__ == "__main__":
    unittest.main()
