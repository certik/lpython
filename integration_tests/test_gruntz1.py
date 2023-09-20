from functools import reduce

from sympy.core import Basic, S, Mul, Pow, PoleError, expand_mul
from sympy.core.cache import cacheit
from sympy.core.numbers import I, oo
from sympy.core.symbol import Dummy, Wild
from sympy.core.traversal import bottom_up

from sympy.functions import log, exp, sign as _sign
from sympy.series.order import Order
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.misc import debug_decorator as debug
from sympy.utilities.timeutils import timethis
from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acot, atan, cos, sin)
from sympy.functions.elementary.complexes import sign as _sign
from sympy.functions.special.error_functions import (Ei, erf)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.polys.polytools import cancel
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.series.gruntz import compare, mrv, rewrite, mrv_leadterm, gruntz, \
    sign
from sympy.testing.pytest import XFAIL, skip, slow

x = Symbol('x', real=True)

class SubsSet(dict):
    """
    Stores (expr, dummy) pairs, and how to rewrite expr-s.

    Explanation
    ===========

    The gruntz algorithm needs to rewrite certain expressions in term of a new
    variable w. We cannot use subs, because it is just too smart for us. For
    example::

        > Omega=[exp(exp(_p - exp(-_p))/(1 - 1/_p)), exp(exp(_p))]
        > O2=[exp(-exp(_p) + exp(-exp(-_p))*exp(_p)/(1 - 1/_p))/_w, 1/_w]
        > e = exp(exp(_p - exp(-_p))/(1 - 1/_p)) - exp(exp(_p))
        > e.subs(Omega[0],O2[0]).subs(Omega[1],O2[1])
        -1/w + exp(exp(p)*exp(-exp(-p))/(1 - 1/p))

    is really not what we want!

    So we do it the hard way and keep track of all the things we potentially
    want to substitute by dummy variables. Consider the expression::

        exp(x - exp(-x)) + exp(x) + x.

    The mrv set is {exp(x), exp(-x), exp(x - exp(-x))}.
    We introduce corresponding dummy variables d1, d2, d3 and rewrite::

        d3 + d1 + x.

    This class first of all keeps track of the mapping expr->variable, i.e.
    will at this stage be a dictionary::

        {exp(x): d1, exp(-x): d2, exp(x - exp(-x)): d3}.

    [It turns out to be more convenient this way round.]
    But sometimes expressions in the mrv set have other expressions from the
    mrv set as subexpressions, and we need to keep track of that as well. In
    this case, d3 is really exp(x - d2), so rewrites at this stage is::

        {d3: exp(x-d2)}.

    The function rewrite uses all this information to correctly rewrite our
    expression in terms of w. In this case w can be chosen to be exp(-x),
    i.e. d2. The correct rewriting then is::

        exp(-w)/w + 1/w + x.
    """
    def __init__(self):
        self.rewrites = {}

    def __repr__(self):
        return super().__repr__() + ', ' + self.rewrites.__repr__()

    def __getitem__(self, key):
        if key not in self:
            self[key] = Dummy()
        return dict.__getitem__(self, key)

    def do_subs(self, e):
        """Substitute the variables with expressions"""
        for expr, var in self.items():
            e = e.xreplace({var: expr})
        return e

    def meets(self, s2):
        """Tell whether or not self and s2 have non-empty intersection"""
        return set(self.keys()).intersection(list(s2.keys())) != set()

    def union(self, s2, exps=None):
        """Compute the union of self and s2, adjusting exps"""
        res = self.copy()
        tr = {}
        for expr, var in s2.items():
            if expr in self:
                if exps:
                    exps = exps.xreplace({var: res[expr]})
                tr[var] = res[expr]
            else:
                res[expr] = var
        for var, rewr in s2.rewrites.items():
            res.rewrites[var] = rewr.xreplace(tr)
        return res, exps

    def copy(self):
        """Create a shallow copy of SubsSet"""
        r = SubsSet()
        r.rewrites = self.rewrites.copy()
        for expr, var in self.items():
            r[expr] = var
        return r

def mmrv(a, b):
    return set(mrv(a, b)[0].keys())

def mrv(e, x):
    """Returns a SubsSet of most rapidly varying (mrv) subexpressions of 'e',
       and e rewritten in terms of these"""
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")
    if not e.has(x):
        return SubsSet(), e
    elif e == x:
        s = SubsSet()
        return s, s[x]
    elif e.is_Mul or e.is_Add:
        i, d = e.as_independent(x)  # throw away x-independent terms
        if d.func != e.func:
            s, expr = mrv(d, x)
            return s, e.func(i, expr)
        a, b = d.as_two_terms()
        s1, e1 = mrv(a, x)
        s2, e2 = mrv(b, x)
        return mrv_max1(s1, s2, e.func(i, e1, e2), x)
    elif e.is_Pow and e.base != S.Exp1:
        e1 = S.One
        while e.is_Pow:
            b1 = e.base
            e1 *= e.exp
            e = b1
        if b1 == 1:
            return SubsSet(), b1
        if e1.has(x):
            base_lim = limitinf(b1, x)
            if base_lim is S.One:
                return mrv(exp(e1 * (b1 - 1)), x)
            return mrv(exp(e1 * log(b1)), x)
        else:
            s, expr = mrv(b1, x)
            return s, expr**e1
    elif isinstance(e, log):
        s, expr = mrv(e.args[0], x)
        return s, log(expr)
    elif isinstance(e, exp) or (e.is_Pow and e.base == S.Exp1):
        # We know from the theory of this algorithm that exp(log(...)) may always
        # be simplified here, and doing so is vital for termination.
        if isinstance(e.exp, log):
            return mrv(e.exp.args[0], x)
        # if a product has an infinite factor the result will be
        # infinite if there is no zero, otherwise NaN; here, we
        # consider the result infinite if any factor is infinite
        li = limitinf(e.exp, x)
        if any(_.is_infinite for _ in Mul.make_args(li)):
            s1 = SubsSet()
            e1 = s1[e]
            s2, e2 = mrv(e.exp, x)
            su = s1.union(s2)[0]
            su.rewrites[e1] = exp(e2)
            return mrv_max3(s1, e1, s2, exp(e2), su, e1, x)
        else:
            s, expr = mrv(e.exp, x)
            return s, exp(expr)
    raise NotImplementedError(
        "Don't know how to calculate the mrv of '%s'" % e)

def mrv_max3(f, expsf, g, expsg, union, expsboth, x):
    """
    Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. max() compares (two elements of)
    f and g and returns either (f, expsf) [if f is larger], (g, expsg)
    [if g is larger] or (union, expsboth) [if f, g are of the same class].
    """
    if not isinstance(f, SubsSet):
        raise TypeError("f should be an instance of SubsSet")
    if not isinstance(g, SubsSet):
        raise TypeError("g should be an instance of SubsSet")
    if f == SubsSet():
        return g, expsg
    elif g == SubsSet():
        return f, expsf
    elif f.meets(g):
        return union, expsboth

    c = compare(list(f.keys())[0], list(g.keys())[0], x)
    if c == ">":
        return f, expsf
    elif c == "<":
        return g, expsg
    else:
        if c != "=":
            raise ValueError("c should be =")
        return union, expsboth

def limitinf(e, x):
    if e == x:
        return oo
    elif e == -x:
        return -oo
    elif isinstance(e, Pow):
        if e.exp.is_Integer:
            exponent = int(e.exp)
            assert exponent != 0
            if exponent > 0:
                return oo
            else:
                return S(0)
        else:
            print(e)
            raise Exception("Pow implemented yet in limitinf().")
    elif isinstance(e, log):
        return oo
    else:
        print(e)
        raise Exception("Not implemented yet.")

def mrv_max1(f, g, exps, x):
    """Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. mrv_max1() compares (two elements of)
    f and g and returns the set, which is in the higher comparability class
    of the union of both, if they have the same order of variation.
    Also returns exps, with the appropriate substitutions made.
    """
    u, b = f.union(g, exps)
    return mrv_max3(f, g.do_subs(exps), g, f.do_subs(exps),
                    u, b, x)

def test_mrv1():
    assert mmrv(x, x) == {x}
    assert mmrv(x + 1/x, x) == {x}
    assert mmrv(x**2, x) == {x}
    assert mmrv(log(x), x) == {x}
    assert mmrv(exp(x), x) == {exp(x)}
    assert mmrv(exp(-x), x) == {exp(-x)}
    assert mmrv(exp(x**2), x) == {exp(x**2)}
    assert mmrv(-exp(1/x), x) == {x}
    #assert mmrv(exp(x + 1/x), x) == {exp(x + 1/x)}

def test_mrv1b():
    assert mmrv(exp(x)+exp(-x), x) == {exp(x), exp(-x)}
    assert mmrv(exp(x)-exp(1/x), x) == {exp(x)}
    assert mmrv(x**2 + x**3, x) == {x}


def test_mrv2a():
    assert mmrv(exp(x + exp(-exp(x))), x) == {exp(-exp(x))}
    assert mmrv(exp(x + exp(-x)), x) == {exp(x + exp(-x)), exp(-x)}
    assert mmrv(exp(1/x + exp(-x)), x) == {exp(-x)}

def test_mrv2b():
    assert mmrv(exp(x + exp(-x**2)), x) == {exp(-x**2)}

def test_mrv2c():
    assert mmrv(
        exp(-x + 1/x**2) - exp(x + 1/x), x) == {exp(x + 1/x), exp(1/x**2 - x)}

def test_mrv3():
    assert mmrv(exp(x**2) + x*exp(x) + log(x)**x/x, x) == {exp(x**2)}
    assert mmrv(
        exp(x)*(exp(1/x + exp(-x)) - exp(1/x)), x) == {exp(x), exp(-x)}
    assert mmrv(log(
        x**2 + 2*exp(exp(3*x**3*log(x)))), x) == {exp(exp(3*x**3*log(x)))}
    assert mmrv(log(x - log(x))/log(x), x) == {x}
    assert mmrv(
        (exp(1/x - exp(-x)) - exp(1/x))*exp(x), x) == {exp(x), exp(-x)}
    assert mmrv(
        1/exp(-x + exp(-x)) - exp(x), x) == {exp(x), exp(-x), exp(x - exp(-x))}
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)), x) == {exp(x*exp(x))}
    assert mmrv(exp(exp(log(log(x) + 1/x))), x) == {x}

def test_mrv4():
    ln = log
    assert mmrv((ln(ln(x) + ln(ln(x))) - ln(ln(x)))/ln(ln(x) + ln(ln(ln(x))))*ln(x),
            x) == {x}
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)) - exp(exp(log(log(x) + 1/x))), x) == \
        {exp(x*exp(x))}


test_mrv1()
test_mrv1b()
