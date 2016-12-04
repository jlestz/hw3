import numpy as N

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        nx = N.size(x)
    except TypeError:
        nx = 1
    fx = f(x)
    try: 
        nf = N.size(fx) 
    except TypeError: 
        nf = 1 
    Df_x = N.matrix(N.zeros((nf,nx)))
    
    # there may be a conflict in x+v if x is a column/row vector
    # with v (below) a row/column vector, so reshape x to match v
    x = N.reshape(N.matrix(x),(nx,1))
    for i in range(nx):
        v = N.matrix(N.zeros((nx,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/dx
    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans

    def Df(self,x): 
        c = self._coeffs
        nc = len(c) 
        ans = 0
        for i in range(nc-1,0,-1): 
            ans = ans + i*c[nc-(i+1)]*x**(i-1)
        return ans 

    def __call__(self, x):
        return self.f(x)

class PolyLog(object):
    """Callable object that is a product of polynomial and logarithm to raised to a nonnegative power

    Example usage: to construct the function f(x) = (x^2 + 2x + 3)*log^3(x),
    and evaluate f(5):

    f = PolyLog([1, 2, 3],3)
    f(5)"""

    def __init__(self, coeffs, power):
        self._poly = Polynomial(coeffs)
        self._power = power
        pcoeffs=N.zeros(power+1)
        pcoeffs[0]=1
        # self._log is the polynomial x^power 
        # will be composed with log in computations
        self._logPolynomial = Polynomial(pcoeffs)

    def __repr__(self):
        polyrepr = repr(self._poly) 
        logrepr = "Log^%s(x)" % str(self._power)
        return polyrepr + logrepr

    def f(self,x):
        flogPolynomial = self._logPolynomial
        fpoly = self._poly 
        return flogPolynomial(N.log(x))*fpoly(x)

    def Df(self,x):
        flogPolynomial = self._logPolynomial
        fpoly = self._poly 

        dflog = flogPolynomial.Df(N.log(x))/x
        dfpoly = fpoly.Df(x)

        return fpoly.f(x)*dflog + dfpoly*flogPolynomial(N.log(x))
    
    def __call__(self, x):
        return self.f(x)

class ExpSin(object):
    """Callable object of the form exp(-(x^2+y^2)*sin(xy) in n dimensions. e.g. call of ExpSin(2) creates the above function in R^2""" 
    
    def __init__(self, dimen):
        self._dimen = dimen

    def __repr__(self):
        return "ExpSin(%s)" % str(self._dimen)

    # can not be called with x as a scalar
    # in 1D, should be called like f([5]) or x =[5], f(x)
    def f(self,x):
        x = N.matrix(x);
        x = N.reshape(x,(self._dimen,1)); 
        sumSq = 0 # sum of squares of variables (argument to exp)
        prod = 1 # product of variables (argument to sin)
        for i in range(self._dimen): 
            sumSq = sumSq + x[i,0]**2 
            prod = prod * x[i,0]

        return N.exp(-sumSq)*N.sin(prod)

    def Df(self,x):
        x = N.matrix(x); 
        x = N.reshape(x,(self._dimen,1)); 
        nx = self._dimen
        Df_x = N.matrix(N.zeros((1,nx)))
        sumSq = 0
        prod = 1
        for i in range(nx): 
            sumSq = sumSq + x[i,0]**2
            prod = prod*x[i,0]
        for i in range(nx): 
            if x[i] > 1e-10: 
                Df_x[0,i] = N.exp(-sumSq)*(-2*x[i,0]*N.sin(prod) + prod*N.cos(prod)/x[i,0])
            # since partial derivative w.r.t x_i has a term like prod/x_i, x_i = 0 case is treated separately
            else: 
                subProd = 1
                for j in range(nx): 
                    if j != i: 
                        prod = prod*x[j,0]
                Df_x[0,i] = N.exp(-sumSq)*(-2*x[i,0]*N.sin(prod) + subProd*N.cos(prod))

        return Df_x
    
    def __call__(self, x):
        return self.f(x)
