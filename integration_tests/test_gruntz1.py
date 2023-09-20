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

def mmrv(a, b):
    return mrv(a, b)

def mrv(e, x):
    """Returns a SubsSet of most rapidly varying (mrv) subexpressions of 'e',
       and e rewritten in terms of these"""
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")
    if not e.has(x):
        return set()
    elif e == x:
        s = {x}
        return s
    elif e.is_Mul or e.is_Add:
        i, d = e.as_independent(x)  # throw away x-independent terms
        if d.func != e.func:
            s, expr = mrv(d, x)
            return s
        a, b = d.as_two_terms()
        s1 = mrv(a, x)
        s2 = mrv(b, x)
        return mrv_max1(s1, s2, e.func(i, e1, e2), x)
    elif e.is_Pow and e.base != S.Exp1:
        e1 = S.One
        while e.is_Pow:
            b1 = e.base
            e1 *= e.exp
            e = b1
        if b1 == 1:
            return set()
        if e1.has(x):
            base_lim = limitinf(b1, x)
            if base_lim is S.One:
                return mrv(exp(e1 * (b1 - 1)), x)
            return mrv(exp(e1 * log(b1)), x)
        else:
            s = mrv(b1, x)
            return s
    elif isinstance(e, log):
        s, expr = mrv(e.args[0], x)
        return s
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
            return s
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
    if f == set():
        return g
    elif g == set():
        return f
    elif f.meets(g):
        return union

    c = compare(list(f.keys())[0], list(g.keys())[0], x)
    if c == ">":
        return f
    elif c == "<":
        return g
    else:
        if c != "=":
            raise ValueError("c should be =")
        return union

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
    print(mmrv(x, x))
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
