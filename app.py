
import sys
from sympy import symbols, simplify, Eq, Interval, sqrt, solve
from sympy.polys.polytools import Poly
from sympy.solvers.inequalities import solve_univariate_inequality
import pandas as pd
import joblib
from pathlib import Path

import sympy as sp
from sympy import (
    symbols, Eq, solve, simplify,
    solveset, S,   
    sqrt, Poly,lambdify
)
import math
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application
)
import os
import tempfile
import webbrowser
import numpy as np
import plotly.graph_objects as go


from sympy import symbols, simplify, Eq, Interval, Union
from sympy.polys.polytools import Poly
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy import solve, sqrt
from sympy import Min, Max

x, y, z = symbols('x y z', real=True)
u = symbols('u', real=True)


transforms = standard_transformations + (implicit_multiplication_application,)
SAFE = {'x': x, 'y': y, 'z': z, 'sin': sp.sin, 'cos': sp.cos,
        'tan': sp.tan, 'sqrt': sp.sqrt, 'Abs': sp.Abs,
        'log': sp.log, 'exp': sp.exp, 'pi': sp.pi, 'E': sp.E}

def parse_equation(s: str) -> Eq:
    s = s.strip().replace('^','**')
    if s.count('=') != 1:
        raise ValueError(f"Need exactly one '=' in {s!r}")
    Ls, Rs = (p.strip() for p in s.split('=',1))
    return Eq(parse_expr(Ls, local_dict=SAFE, transformations=transforms),
              parse_expr(Rs, local_dict=SAFE, transformations=transforms))


def _preprocess_eq(eq: sp.Eq) -> sp.Eq:
    """
    If either side contains a sqrt(...), square both sides so we get a pure
    polynomial equation.  Otherwise return eq unchanged.
    """
    # collect all Pow-atoms (factors) in lhs and rhs
    pows = list(eq.lhs.atoms(sp.Pow)) + list(eq.rhs.atoms(sp.Pow))
    # check if any has exponent 1/2 (i.e. sqrt)
    if any(f.exp == sp.Rational(1,2) for f in pows):
        # square both sides: (lhs)**2 = (rhs)**2
        return sp.Eq(eq.lhs**2, eq.rhs**2)
    return eq

# ─── Improved region_type ─────────────────────────────────────────────────────
def region_type(eq: sp.Eq) -> str:
    eq = _preprocess_eq(eq)
    expr = eq.lhs - eq.rhs
    poly = sp.Poly(expr, x, y, z)
    deg = poly.total_degree()

    # 1) Sphere: deg=2, all three x²,y²,z² coeffs equal & nonzero, no cross/linear
    c_x2 = poly.coeff_monomial(x**2)
    c_y2 = poly.coeff_monomial(y**2)
    c_z2 = poly.coeff_monomial(z**2)
    crosses = sum(poly.coeff_monomial(m) for m in [(1,1,0),(1,0,1),(0,1,1)])
    linears = sum(poly.coeff_monomial(m) for m in [(1,0,0),(0,1,0),(0,0,1)])
    c0      = poly.coeff_monomial(1)
    if (deg==2
        and c_x2==c_y2==c_z2!=0
        and crosses==0
        and linears==0):
        return 'sphere'

    # 2) Cylinder: deg=2, no cross‐terms, no linear terms, **exactly one** of {x²,y²,z²} is zero
    if (deg==2
         and crosses==0
         and linears==0):
         nz = [v for v in (c_x2, c_y2, c_z2) if v!=0]
         zc = [v for v in (c_x2, c_y2, c_z2) if v==0]
         if len(nz)==2 and len(zc)==1:
             return 'cylinder'
   

    # 3) Paraboloid: deg=2, linear in z, at least one of x² or y² present
    if deg==2 and poly.degree(z)==1 and any(poly.coeff_monomial(m)!=0 for m in [(2,0,0),(0,2,0)]):
        return 'paraboloid'

    # 4) Cone: deg=2, quadratic in z, **no** constant term
    if deg==2 and poly.degree(z)==2:
        if crosses==0 and poly.coeff_monomial((1,0,0))==0 and poly.coeff_monomial((0,1,0))==0:
            return 'cone'

    # 5) Plane: deg=1
    if deg==1:
        return 'plane'
    raise ValueError(f"Cannot identify region type for {eq}")


def handle_sphere_cylinder(eqA, eqB):
    # ─── 0) Auto-detect & swap so sph_eq is sphere, cyl_eq is cylinder ────────
    from math import isclose
    def is_sphere(eq):
        p = Poly(eq.lhs - eq.rhs, x, y, z)
        # must have x^2,y^2,z^2 all same coeff, total degree=2, and no linear or cross terms
        return (p.total_degree()==2
                and p.coeff_monomial(x**2)==p.coeff_monomial(y**2)==p.coeff_monomial(z**2)
                and all(p.coeff_monomial(mon)==0
                        for mon in [(1,1,0),(1,0,1),(0,1,1),(1,0,0),(0,1,0),(0,0,1)]))
    if is_sphere(eqA):
        sph_eq, cyl_eq = eqA, eqB
    else:
        sph_eq, cyl_eq = eqB, eqA

    sph_eq = _preprocess_eq(sph_eq)  
    # ─── 1) Extract R² from sphere:  c*(x²+y²+z²) + c0 = 0 --> R2 = -c0/c ─────
    Ps   = Poly(sph_eq.lhs - sph_eq.rhs, x, y, z)
    c_s  = Ps.coeff_monomial(x**2)
    c0_s = Ps.coeff_monomial(1)
    R2   = simplify(-c0_s / c_s)
    cyl_eq = _preprocess_eq(cyl_eq)
    # ─── 2) Extract cylinder:  a*x² + b*y² + c0 = 0 --> a x² + b y² = C ─────
    Pc   = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)  # ignore z
    a    = Pc.coeff_monomial(x**2)
    b    = Pc.coeff_monomial(y**2)
    c0_c = Pc.coeff_monomial(1)
    C    = simplify(-c0_c)

    # ─── 3) Build the limits ─────────────────────────────────────────────────
    # x : –√(C/a) … +√(C/a)
    x_lo = -sqrt(C/a)
    x_hi =  sqrt(C/a)

    # y : –√((C – a x²)/b) … +√((C – a x²)/b)
    y_lo = -sqrt((C - a*x**2)/b)
    y_hi =  sqrt((C - a*x**2)/b)

    # z : –√(R² – x² – y²) … +√(R² – x² – y²)
    z_lo = -sqrt(R2 - x**2 - y**2)
    z_hi =  sqrt(R2 - x**2 - y**2)

    return simplify(x_lo), simplify(x_hi), \
           simplify(y_lo), simplify(y_hi), \
           simplify(z_lo), simplify(z_hi)
# ─── Corrected plane-plane handler ────────────────────────────────────────────
def handle_plane_plane(eq1, eq2, constraints=None):
    """
    Analytic dz dy dx limits for the bounded wedge formed by two planes
    plus any extra x,y half-plane constraints.
    `constraints` should be a list of sympy Relational objects in x,y,
    e.g. [x>=0, y>=0, x+y<=2].
    """
    # 1) Solve each for z = f(x,y)
    z1 = solve(eq1, z)[0]
    z2 = solve(eq2, z)[0]

    # 2) Order them so z1 is the lower surface at (0,0)
    if float(z1.subs({x:0,y:0})) > float(z2.subs({x:0,y:0})):
        z1, z2 = z2, z1

    # 3) Project eq1 to z=0 →  A x + B y + C = 0
    proj = eq1.lhs - eq1.rhs
    proj0 = sp.expand(proj.subs(z, 0))
    A = proj0.coeff(x)
    B = proj0.coeff(y)
    C = -proj0.subs({x:0,y:0})   # so A x + B y = C

    # 4) Compute axis intercepts and auto-pick half-planes
    #    x-intercept: y=0 → x0 = C/A
    #    y-intercept: x=0 → y0 = C/B
    x0 = simplify(C/A)
    y0 = simplify(C/B)

    # Decide which side of each axis is valid:
    # test point (0,0) must satisfy the half-planes
    def side(rel, pt):
        # rel is e.g. x >= 0 or x+y <= 2
        return bool(rel.subs({x:pt[0], y:pt[1]}))

    # Default constraints: from axes
    if constraints is None:
        c1 = x >= 0 if side(x>=0,(x0/2,0)) else x <= 0
        c2 = y >= 0 if side(y>=0,(0,y0/2)) else y <= 0
        constraints = [c1, c2]

    # Always include the plane-bound half-planes:
    constraints = constraints + [
        A*x + B*y <= C,   # projection of eq1
        # optionally also include eq2’s projection if you want the other cut
        # B2*x + D*y <= E
    ]

    # 5) Build the polygon’s vertices by intersecting each pair of boundary lines
    lines = []
    # convert each constraint to a line equation A*x+B*y=C if it’s an eq,
    # or keep as inequality; for intersection purposes we turn >= into = for the border
    for con in constraints:
        if con.is_Relational:
            lhs, rhs = con.lhs, con.rhs
            # border line: lhs - rhs = 0
            lines.append(lhs - rhs)
        else:
            raise ValueError("Constraints must be relational x,y inequalities")

    verts = set()
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            sol = solve([lines[i], lines[j]], (x,y), dict=True)
            if sol:
                xi, yi = sol[0][x], sol[0][y]
                # check every half-plane
                if all(bool(c.subs({x:xi,y:yi})) for c in constraints):
                    verts.add((xi, yi))
    verts = sorted(verts)

    if len(verts) < 3:
        raise ValueError("Region not bounded by given constraints")

    # 6) Derive from–to limits
    x_lo, x_hi = min(v[0] for v in verts), max(v[0] for v in verts)
    # y from its lower border (solve each constraint for y) up to the top of the polygon
    def y_top(xv):
        # among all border lines of form A*x + B*y = C, pick the minimal positive root
        ys = []
        for line in lines:
            # line = LHS(x,y) = 0  →  B*y = -A*x + C
            if line.has(y):
                Ai = line.coeff(x,1)
                Bi = line.coeff(y,1)
                Ci = -line.subs({x:0,y:0})
                ys.append((Ci - Ai*xv)/Bi)
        return min(ys)
    y_lo = 0  # because we used x>=0,y>=0
    y_hi = lambda xv: simplify(y_top(xv))

    # 7) Return in dz dy dx order
    return (
      simplify(x_lo), simplify(x_hi),
      lambda xv: simplify(y_lo), lambda xv: simplify(y_hi(xv)),
      simplify(z1), simplify(z2)
    )
# ─── Register it ───────────────────────────────────────────────────────────────
def handle_sphere_paraboloid(sph_eq, para_eq):
    # ─── 1) Extract R² from sphere ───────────────────────────────
    # solve s.t. z**2 = R2 - x^2 - y^2
    sol_z2 = solve(sph_eq, z**2)
    if not sol_z2:
        raise ValueError("Can't isolate z² in the sphere equation.")
    R2 = simplify(sol_z2[0].subs({x:0, y:0}))

    # ─── 2) Isolate paraboloid z = a*(x^2+y^2) + c ───────────────
    para_sols = solve(para_eq, z)
    if len(para_sols)!=1:
        raise ValueError("Paraboloid must be linear in z.")
    z_para = simplify(para_sols[0])

    # check it’s of the form a·(x²+y²)+c
    Pxy = Poly(z_para, x, y)
    mons = set(Pxy.monoms())
    if mons - {(2,0),(0,2),(0,0)}:
        raise ValueError("Paraboloid is not circular in x²+y².")
    a1 = Pxy.coeff_monomial(x**2)
    a2 = Pxy.coeff_monomial(y**2)
    c0 = Pxy.coeff_monomial(1)
    if simplify(a1 - a2)!=0:
        raise ValueError("Not a *circular* paraboloid (a_x≠a_y).")
    a = simplify(a1)
    c = simplify(c0)

    # ─── 3) Build the u-quadratic: (a·u + c)² + u – R² = 0 ─────────
    u = symbols('u', real=True)
    quad = Poly((a*u + c)**2 + u - R2, u)
    roots = solve(quad.as_expr(), u)
    # pick the positive real root
    up = [s for s in roots if s.is_real and s>0]
    if not up:
        raise ValueError("No valid positive intersection radius.")
    u_pos = simplify(up[0])

    # ─── 4) Form the limits ───────────────────────────────────────
    x_lo, x_hi = -sqrt(u_pos),  sqrt(u_pos)
    y_lo = -sqrt(u_pos - x**2)
    y_hi =  sqrt(u_pos - x**2)
    z_lo = z_para
    z_hi = sqrt(R2 - x**2 - y**2)

    return simplify(x_lo), simplify(x_hi), \
           simplify(y_lo), simplify(y_hi), \
           simplify(z_lo), simplify(z_hi)

def handle_cylinder_paraboloid(eqA, eqB):
    # ─── 1) Auto-swap ──────────────────────────────────────────────
    if region_type(eqA) == 'cylinder':
        cyl_eq, para_eq = eqA, eqB
    elif region_type(eqB) == 'cylinder':
        cyl_eq, para_eq = eqB, eqA
    else:
        raise ValueError("Neither equation is a cylinder")

    # ─── 2) Extract cylinder: a x² + b y² = C ──────────────────────
    P = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = P.coeff_monomial(x**2)
    b = P.coeff_monomial(y**2)
    c0 = P.coeff_monomial(1)
    C  = simplify(-c0)  # so that a x² + b y² = C

    # ─── 3) Extract paraboloid: z = f(x,y) ────────────────────────
    sols = solve(para_eq, z)
    if len(sols) != 1:
        raise ValueError("Paraboloid must give exactly one z-solution")
    f = simplify(sols[0])  # f(x,y) = a(x²+y²) + c

    # ─── 4) x and y bounds from the cylinder ──────────────────────
    x_lo = -sqrt(C/a)
    x_hi =  sqrt(C/a)
    y_lo = -sqrt((C - a*x**2)/b)
    y_hi =  sqrt((C - a*x**2)/b)

    # ─── 5) z bounds between the xy-plane and the paraboloid ────
    #    we take z=0 plane as base, so region is {0 ≤ z ≤ f(x,y)} or vice versa
    base = 0
    top  = f
    # ensure base < top at a sample point (0,0)
    if (top.subs({x:0,y:0}) < base):
        base, top = top, base

    return (simplify(x_lo), simplify(x_hi),
            simplify(y_lo), simplify(y_hi),
            simplify(base),  simplify(top))

# def handle_paraboloid_paraboloid(eq1, eq2):
#     z1 = solve(eq1, z)[0]
#     z2 = solve(eq2, z)[0]
#     # intersection curve z1 = z2 -> x^2+y^2 = const
#     R2 = sp.solve(z1 - z2, x**2 + y**2)[0]
#     R = sp.sqrt(R2)
#     z_lo, z_hi = sp.Min(z1, z2), sp.Max(z1, z2)
#     y_lo = -sp.sqrt(R**2 - x**2)
#     y_hi =  sp.sqrt(R**2 - x**2)
#     x_lo, x_hi = -R, R
#     return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi


def handle_paraboloid_paraboloid(eq1, eq2):
    """
    Intersection of two circular paraboloids:
      eq1: z = f1(x,y) and eq2: z = f2(x,y).
    Returns (x_lo,x_hi, y_lo,y_hi, z_lo,z_hi).
    """
    # 1) Solve for z in each
    sols1 = solve(eq1, z)
    sols2 = solve(eq2, z)
    if len(sols1) != 1 or len(sols2) != 1:
        raise ValueError("Each paraboloid must give exactly one z=f(x,y).")
    z1 = simplify(sols1[0])
    z2 = simplify(sols2[0])

    # 2) Δ = z1 - z2 = 0 => a·x² + a·y² + c0 = 0
    Δ = simplify(z1 - z2)
    P = Poly(Δ, x, y)
    a = P.coeff_monomial(x**2)
    if simplify(P.coeff_monomial(y**2) - a) != 0:
        raise ValueError("Paraboloids must share the same x²,y² coefficient.")
    c0 = P.coeff_monomial(1)

    # 3) Compute R² = -c0/a
    if a == 0:
        raise ValueError("Intersection is not circular.")
    R2 = simplify(-c0 / a)
    R  = sqrt(R2)

    # 4) Determine which z is lower at (0,0)
    z0_1 = float(z1.subs({x:0,y:0}))
    z0_2 = float(z2.subs({x:0,y:0}))
    if z0_1 < z0_2:
        z_lo_sym, z_hi_sym = z1, z2
    else:
        z_lo_sym, z_hi_sym = z2, z1

    # 5) Build and return the limits
    x_lo = -R
    x_hi =  R
    y_lo = -sqrt(R2 - x**2)
    y_hi =  sqrt(R2 - x**2)
    return ( simplify(x_lo), simplify(x_hi),
             simplify(y_lo), simplify(y_hi),
             simplify(z_lo_sym), simplify(z_hi_sym) )

def handle_plane_plane(eq1, eq2):
    # 1) Solve each for z
    z1 = solve(eq1, z)[0]
    z2 = solve(eq2, z)[0]

    # 2) Pick which is lower/higher at (0,0)
    if z1.subs({x:0, y:0}) < z2.subs({x:0, y:0}):
        z_lo, z_hi = z1, z2
    else:
        z_lo, z_hi = z2, z1

    # 3) Project to z=0: extract A,B,C so A*x + B*y = C
    expr1 = simplify((eq1.lhs - eq1.rhs).subs(z, 0))
    A = expr1.coeff(x)
    B = expr1.coeff(y)
    K = expr1.subs({x:0, y:0})  # expr1 = A*x + B*y + K = 0
    C = -K                     # so A*x + B*y = C

    # 4) x,y bounds from axes + that line
    x_lo, x_hi = 0, simplify(C/A)
    y_lo       = 0
    y_hi       = simplify((C - A*x)/B)

    return x_lo, x_hi, y_lo, y_hi, simplify(z_lo), simplify(z_hi)

def handle_sphere_cone(sph_eq, cone_eq):
    # Extract R^2 from sphere eq: x^2+y^2+z^2 = R^2
    P = sp.Poly(sph_eq.lhs - sph_eq.rhs, x, y, z)
    c0  = P.coeff_monomial(1)
    c_x2 = P.coeff_monomial(x**2)
    R2 = -c0/c_x2

    # For cone z^2 = x^2+y^2 we know the intersection circle has radius^2 = R2/2
    r2 = R2/2

    x_lo, x_hi = -sp.sqrt(r2),  sp.sqrt(r2)
    y_lo        = -sp.sqrt(r2 - x**2)
    y_hi        =  sp.sqrt(r2 - x**2)

    z_lo =  sp.sqrt(x**2 + y**2)
    z_hi =  sp.sqrt(R2 - x**2 - y**2)

    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi

def handle_cylinder_cone(eqA, eqB):
    # auto-swap so cyl_eq is the cylinder, cone_eq the cone
    if region_type(eqA) == 'cylinder':
        cyl_eq, cone_eq = eqA, eqB
    else:
        cyl_eq, cone_eq = eqB, eqA

    # 1) Extract cylinder: a*x^2 + b*y^2 + … = 0  ⇒  a x^2 + b y^2 = C
    Pc   = sp.Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a    = Pc.coeff_monomial(x**2)
    b    = Pc.coeff_monomial(y**2)
    c0   = Pc.coeff_monomial(1)
    C    = simplify(-c0)

    # 2) Extract cone: z^2 = Q(x,y)
    sol_z2 = solve(cone_eq, z**2)
    if not sol_z2:
        raise ValueError("Cannot isolate z^2 from the cone equation")
    Q = simplify(sol_z2[0])

    # 3) Build the dz dy dx limits
    x_lo, x_hi = -sqrt(C/a), sqrt(C/a)
    y_lo, y_hi = -sqrt((C - a*x**2)/b), sqrt((C - a*x**2)/b)
    z_lo, z_hi = -sqrt(Q), sqrt(Q)

    return simplify(x_lo), simplify(x_hi), \
           simplify(y_lo), simplify(y_hi), \
           simplify(z_lo), simplify(z_hi)

def handle_cylinder_cylinder(eq1, eq2):
    # ─── 1) Figure out which is the "vertical" cylinder (no z-term)
    P1 = Poly(eq1.lhs - eq1.rhs, x, y, z)
    P2 = Poly(eq2.lhs - eq2.rhs, x, y, z)
    if P1.coeff_monomial(z**2) == 0 and (P1.coeff_monomial(x**2) or P1.coeff_monomial(y**2)):
        vert_eq, perp_eq = eq1, eq2
        Pv, Pp = P1, P2
    elif P2.coeff_monomial(z**2) == 0 and (P2.coeff_monomial(x**2) or P2.coeff_monomial(y**2)):
        vert_eq, perp_eq = eq2, eq1
        Pv, Pp = P2, P1
    else:
        raise ValueError("Need exactly one cylinder with no z² term")

    # ─── 2) Extract vertical‐cylinder:  a x² + b y² = C
    a = Pv.coeff_monomial(x**2)
    b = Pv.coeff_monomial(y**2)
    C = simplify(-Pv.coeff_monomial(1))

    # ─── 3) Extract perpendicular‐cylinder: either y²+z² or x²+z²
    #    so we can solve for z² = (C2 – α·var²)/β
    p_y2 = Pp.coeff_monomial(y**2)
    p_z2 = Pp.coeff_monomial(z**2)
    p_x2 = Pp.coeff_monomial(x**2)
    C2   = simplify(-Pp.coeff_monomial(1))

    if p_y2 and p_z2:
        # cylinder axis ∥ x-axis:  y²+z² = C2
        alpha, beta = p_y2, p_z2
        z_lo = -sqrt((C2 - alpha*y**2)/beta)
        z_hi =  sqrt((C2 - alpha*y**2)/beta)
    elif p_x2 and p_z2:
        # cylinder axis ∥ y-axis:  x²+z² = C2
        alpha, beta = p_x2, p_z2
        z_lo = -sqrt((C2 - alpha*x**2)/beta)
        z_hi =  sqrt((C2 - alpha*x**2)/beta)
    else:
        raise ValueError("Perp cylinder must involve z² + exactly one of x² or y²")

    # ─── 4) Build the dz dy dx limits
    x_lo = -sqrt(C/a)
    x_hi =  sqrt(C/a)

    y_lo = -sqrt((C - a*x**2)/b)
    y_hi =  sqrt((C - a*x**2)/b)

    return ( simplify(x_lo), simplify(x_hi),
             simplify(y_lo), simplify(y_hi),
             simplify(z_lo), simplify(z_hi) )

def handle_cone_paraboloid(eqA, eqB):
    # 1) auto-swap
    if region_type(eqA) == 'cone':
        cone_eq, para_eq = eqA, eqB
    elif region_type(eqB) == 'cone':
        cone_eq, para_eq = eqB, eqA
    else:
        raise ValueError("Need one cone & one paraboloid")

    # 2) isolate paraboloid: z = a*(x^2+y^2) + c
    psol = solve(para_eq, z)
    if len(psol) != 1:
        raise ValueError("Paraboloid must give one z=…")
    z_para = simplify(psol[0])
    Pp = Poly(z_para, x, y)
    mons = set(Pp.monoms())
    if not mons <= {(2,0),(0,2),(0,0)}:
        raise ValueError("Paraboloid not circular in x^2+y^2")
    a = simplify(Pp.coeff_monomial(x**2))
    if simplify(Pp.coeff_monomial(y**2) - a) != 0:
        raise ValueError("Paraboloid must have same x^2,y^2 coeff")
    c = simplify(Pp.coeff_monomial(1))

    # 3) isolate cone: z^2 = A*(x^2+y^2)
    csol = solve(cone_eq, z**2)
    if not csol:
        raise ValueError("Cannot isolate z^2 from cone")
    Q = simplify(csol[0])
    Pc = Poly(Q, x, y)
    mons_c = set(Pc.monoms())
    if not mons_c <= {(2,0),(0,2)}:
        raise ValueError("Cone not circular: z^2 ≠ A*(x^2+y^2)")
    A = simplify(Pc.coeff_monomial(x**2))
    if simplify(Pc.coeff_monomial(y**2) - A) != 0:
        raise ValueError("Cone must have same x^2,y^2 coeff")

    # 4) solve (a u + c)^2 = A u  →  a^2 u^2 + (2ac - A) u + c^2 = 0
    quad = simplify(a**2*u**2 + (2*a*c - A)*u + c**2)
    roots = solve(quad, u)
    us = [r for r in roots if r.is_real and r > 0]
    if not us:
        raise ValueError("No valid intersection radius found")
    u0 = simplify(us[0])

    # 5) build bounds
    x_lo, x_hi = -sqrt(u0), sqrt(u0)
    y_lo, y_hi = -sqrt(u0 - x**2), sqrt(u0 - x**2)

    # decide which is lower at (0,0)
    z0_para = c
    z0_cone = 0
    if z0_para <= z0_cone:
        z_lo, z_hi = z_para,  sqrt(A*(x**2 + y**2))
    else:
        z_lo, z_hi = -sqrt(A*(x**2 + y**2)), z_para

    return simplify(x_lo), simplify(x_hi), \
           simplify(y_lo), simplify(y_hi), \
           simplify(z_lo), simplify(z_hi)

def handle_cylinder_plane(eqA, eqB):
    # 1) Identify cylinder vs. plane
    if region_type(eqA) == 'cylinder':
        cyl_eq, plane_eq = eqA, eqB
    elif region_type(eqB) == 'cylinder':
        cyl_eq, plane_eq = eqB, eqA
    else:
        raise ValueError("Need one cylinder and one plane")

    # 2) Extract cylinder: a*x^2 + b*y^2 = C
    Pc   = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a    = Pc.coeff_monomial(x**2)
    b    = Pc.coeff_monomial(y**2)
    C    = simplify(-Pc.coeff_monomial(1))

    # 3) Build x,y bounds
    x_lo = -sqrt(C/a)
    x_hi =  sqrt(C/a)
    y_lo = -sqrt((C - a*x**2)/b)
    y_hi =  sqrt((C - a*x**2)/b)

    # 4) Solve plane for z
    if z not in plane_eq.free_symbols:
        raise ValueError("Plane must contain z")
    z_sols = solve(plane_eq, z)
    if len(z_sols) != 1:
        raise ValueError("Plane must give exactly one z=…")
    z_plane = simplify(z_sols[0])

    # 5) Use [0, z_plane] as z-bounds, swapping if needed
    z_lo, z_hi = 0, z_plane
    if float(z_hi.subs({x:0,y:0})) < float(z_lo):
        z_lo, z_hi = z_hi, z_lo

    return ( simplify(x_lo), simplify(x_hi),
             simplify(y_lo), simplify(y_hi),
             simplify(z_lo), simplify(z_hi) )

from sympy import symbols, Poly, solve, sqrt as sp_sqrt, simplify
import sympy as sp
def handle_sphere_plane(eq1, eq2):
    x,y,z = symbols('x y z', real=True)

    # 1) Identify sphere vs plane
    def _is_sphere(e):
        P = Poly(e.lhs - e.rhs, x,y,z)
        c2 = P.coeff_monomial(x**2)
        return (P.total_degree()==2
                and c2!=0
                and P.coeff_monomial(y**2)==c2
                and P.coeff_monomial(z**2)==c2
                and all(P.coeff_monomial(m)==0
                        for m in [(1,0,0),(0,1,0),(0,0,1),
                                  (1,1,0),(1,0,1),(0,1,1)]))
    if _is_sphere(eq1):
        sph_eq, plane_eq = eq1, eq2
    else:
        sph_eq, plane_eq = eq2, eq1

    # 2) Extract R^2
    sol_z2 = solve(sph_eq, z**2)
    if not sol_z2:
        raise ValueError("Couldn't isolate z² in sphere.")
    R2 = simplify(sol_z2[0].subs({x:0,y:0}))

    # 3) Plane coefficients A x + B y + C z = D
    expr = plane_eq.lhs - plane_eq.rhs
    Pp = Poly(expr, x,y,z)
    A = Pp.coeff_monomial(x)
    B = Pp.coeff_monomial(y)
    C = Pp.coeff_monomial(z)
    D = -Pp.coeff_monomial(1)

    # 4) Vertical planes (C == 0)
    if C == 0:
        if B==0 and A!=0:
            k = simplify(D/A)
            x_lo, x_hi = k, sp_sqrt(R2)
            y_lo = -sp_sqrt(R2 - x**2)
            y_hi =  sp_sqrt(R2 - x**2)
        elif A==0 and B!=0:
            k = simplify(D/B)
            y_lo, y_hi = k, sp_sqrt(R2)
            x_lo = -sp_sqrt(R2 - y**2)
            x_hi =  sp_sqrt(R2 - y**2)
        else:
            y_line = simplify((D - A*x)/B)
            disc = simplify(R2 - x**2 - y_line**2)
            xs = solve(disc, x)
            reals = sorted([s for s in xs if s.is_real],
                           key=lambda s: float(s))
            if len(reals)<2:
                raise ValueError("No real x‐bounds for vertical plane.")
            x_lo, x_hi = simplify(reals[0]), simplify(reals[-1])
            y_lo = y_line
            y_hi =  sp_sqrt(R2 - x**2)
        z_lo = -sp_sqrt(R2 - x**2 - y**2)
        z_hi =  sp_sqrt(R2 - x**2 - y**2)
        return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi

    # 5) Slanted/horizontal: z = (D - A*x - B*y)/C
    z_plane = simplify((D - A*x - B*y)/C)

    # 6) Intersection in xy: x^2+y^2+z_plane^2 = R2
    E = simplify(x**2 + y**2 + z_plane**2 - R2)

    # 7) x‐bounds from E(x,0)=0
    xs0 = solve(E.subs(y,0), x)
    reals_x = sorted([s for s in xs0 if s.is_real],
                     key=lambda s: float(s))
    if len(reals_x)!=2:
        raise ValueError("Couldn't find two real x‐bounds.")
    x_lo, x_hi = simplify(reals_x[0]), simplify(reals_x[1])

    # 8) y‐bounds at fixed x
    Ey = Poly(E, y)
    Ay = Ey.coeff_monomial(y**2)
    By = Ey.coeff_monomial(y)
    Cy = Ey.coeff_monomial(1)
    disc = simplify(By**2 - 4*Ay*Cy)
    sd = sp_sqrt(disc)
    y_lo = simplify((-By - sd)/(2*Ay))
    y_hi = simplify((-By + sd)/(2*Ay))

    # 9) z‐bounds ordered by Min/Max
    z_sph = sp_sqrt(R2 - x**2 - y**2)
    z_lo = simplify(sp.Min(z_plane, z_sph))
    z_hi = simplify(sp.Max(z_plane, z_sph))

    return simplify(x_lo), simplify(x_hi), \
           simplify(y_lo), simplify(y_hi), \
           simplify(z_lo), simplify(z_hi)
# def handle_paraboloid_plane(eqA, eqB):
#     """
#     Given one paraboloid (z = f(x,y)) and one plane (C*z + A*x + B*y = D),
#     returns (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) for dz dy dx integration.
#     Uses find_z_bounds and find_xy_bounds to auto‐solve the intersection.
#     """
#     # 1) Identify which is which
#     if region_type(eqA) == 'paraboloid':
#         para_eq, plane_eq = eqA, eqB
#     elif region_type(eqB) == 'paraboloid':
#         para_eq, plane_eq = eqB, eqA
#     else:
#         raise ValueError("Need one paraboloid and one plane")

#     # 2) Both are z‐surfaces; no side‐constraints
#     surfs = [para_eq, plane_eq]
#     cons = []

#     # 3) Compute the two z‐bounds
#     z_lo, z_hi = find_z_bounds(surfs)

#     # 4) Compute x,y bounds from their intersection curve
#     (y_lo, y_hi), (x_lo, x_hi) = find_xy_bounds(surfs, cons, (z_lo, z_hi))

#     # 5) Return tidy symbolic limits
#     return ( simplify(x_lo), simplify(x_hi),
#              simplify(y_lo), simplify(y_hi),
#              simplify(z_lo), simplify(z_hi) )


# def handle_paraboloid_plane(eqA, eqB):
#     """
#     Intersection of a circular paraboloid (z = f(x,y))
#     with a plane (z = g(x,y)). Returns
#       (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
#     for ∫∫∫ dz dy dx.
#     """
#     x, y, z = symbols('x y z', real=True)

#     # 1) Identify which is which
#     if region_type(eqA) == 'paraboloid':
#         para_eq, plane_eq = eqA, eqB
#     else:
#         para_eq, plane_eq = eqB, eqA

#     # 2) Paraboloid: solve z = f(x,y)
#     psol = solve(para_eq, z)
#     if len(psol) != 1:
#         raise ValueError("Paraboloid must yield exactly one z=f(x,y).")
#     z_para = simplify(psol[0])

#     # 3) Plane: must solve z = g(x,y)
#     if z not in plane_eq.free_symbols:
#         raise ValueError("Plane must be of the form z = g(x,y).")
#     z_plane = simplify(solve(plane_eq, z)[0])

#     # 4) Intersection curve: C(x,y) = f(x,y) - g(x,y) = 0
#     C = simplify(z_para - z_plane)

#     # 5) Solve C=0 for y → two branches y1(x), y2(x)
#     y_sols = solve(C, y)
#     if len(y_sols) != 2:
#         raise ValueError("Expected two y‐branches from intersection.")
#     y1, y2 = [simplify(sol) for sol in y_sols]

#     # 6) Build the discriminant of C viewed as a quadratic in y:
#     poly = Poly(C, y)
#     D = simplify(poly.discriminant())

#     # 7) Solve D(x) = 0 over ℝ to find the two x‐endpoints
#     x_roots = solveset(Eq(D, 0), x, domain=S.Reals)
#     if not hasattr(x_roots, "__iter__") or len(x_roots) < 2:
#         raise ValueError("Couldn't find two real x‐bounds from discriminant")
#     x_vals = sorted(x_roots, key=lambda v: float(v))
#     x_lo, x_hi = map(simplify, (x_vals[0], x_vals[-1]))

#     # 8) Order the two y‐branches by sampling at x_mid
#     x_mid = (x_lo + x_hi) / 2
#     y1m = float(y1.subs(x, x_mid))
#     y2m = float(y2.subs(x, x_mid))
#     if y1m < y2m:
#         y_lo, y_hi = y1, y2
#     else:
#         y_lo, y_hi = y2, y1

#     # 9) Pick z‐bounds by sampling the same midpoint
#     y_mid = float((y_lo + y_hi).subs(x, x_mid) / 2)
#     sample = {x: float(x_mid), y: y_mid}
#     if float(z_plane.subs(sample)) < float(z_para.subs(sample)):
#         z_lo, z_hi = z_plane, z_para
#     else:
#         z_lo, z_hi = z_para, z_plane

#     return (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)





def handle_paraboloid_plane(eqA, eqB):
    """
    Intersection of a *circular* paraboloid z=f(x,y)
    with *any* single plane eqB (either z=g(x,y) or vertical).
    Returns (x_lo,x_hi, y_lo,y_hi, z_lo,z_hi) for ∫ dz dy dx.
    """
    x, y, z = symbols('x y z', real=True)

    # 1) Identify paraboloid vs plane
    if region_type(eqA) == 'paraboloid':
        para_eq, plane_eq = eqA, eqB
    else:
        para_eq, plane_eq = eqB, eqA

    # 2) Extract paraboloid: z = f(x,y)
    z_sols = solve(para_eq, z)
    if len(z_sols) != 1:
        raise ValueError("Paraboloid must give exactly one z = f(x,y).")
    z_para = simplify(z_sols[0])

    # 3) Check if plane involves z
    if z in plane_eq.free_symbols:
        # ── CASE I: plane is z = g(x,y) ───────────────────────────────
        z_plane = simplify(solve(plane_eq, z)[0])

        # Intersection curve: f(x,y)=g(x,y) → C(x,y)=0
        C = simplify(z_para - z_plane)

        # Quadratic in y:
        P = Poly(C, y)
        A = P.coeff_monomial(y**2)
        B = P.coeff_monomial(y**1)
        C0 = P.coeff_monomial(1)
        D = simplify(B**2 - 4*A*C0)

        # x‐bounds from D=0
        x_roots = solveset(Eq(D, 0), x, domain=S.Reals)
        x_list = sorted([r for r in x_roots if r.is_real], key=lambda v: float(v))
        if len(x_list) < 2:
            raise ValueError("Couldn't find two real x‐bounds from discriminant.")
        x_lo, x_hi = x_list[0], x_list[-1]

        # y‐branches
        y_br = solve(C, y)
        if len(y_br) != 2:
            raise ValueError("Expected two y-branches from intersection.")
        y1, y2 = simplify(y_br[0]), simplify(y_br[1])

        # order by sampling
        x_mid = (x_lo + x_hi)/2
        if float(y1.subs(x, x_mid)) < float(y2.subs(x, x_mid)):
            y_lo, y_hi = y1, y2
        else:
            y_lo, y_hi = y2, y1

        # z‐bounds by sampling inside
        y_mid = float((y_lo + y_hi).subs(x, x_mid) / 2)
        pt = {x: float(x_mid), y: y_mid}
        if float(z_plane.subs(pt)) < float(z_para.subs(pt)):
            z_lo, z_hi = z_plane, z_para
        else:
            z_lo, z_hi = z_para, z_plane

        return (simplify(x_lo), simplify(x_hi),
                simplify(y_lo), simplify(y_hi),
                simplify(z_lo), simplify(z_hi))

    else:
        # ── CASE II: plane is vertical (no z) → side constraint ────────
        # We'll take z from 0 up to the paraboloid
        z_lo, z_hi = 0, z_para

        # One z‐surface + one xy‐constraint
        # Represent the z‐surface for find_xy_bounds as Eq(z, z_para)
        surf = Eq(z, z_para)
        cons = [plane_eq]

        # find_xy_bounds knows how to handle one surf + one cons
        (y_lo, y_hi), (x_lo, x_hi) = find_xy_bounds([surf], cons, (z_lo, z_hi))

        return (simplify(x_lo), simplify(x_hi),
                simplify(y_lo), simplify(y_hi),
                simplify(z_lo), simplify(z_hi))
def handle_sphere_sphere(eq1, eq2):
    # 1) Extract R² from each sphere by isolating z² = … 
    s1 = solve(eq1, z**2)
    s2 = solve(eq2, z**2)
    if not s1 or not s2:
        raise ValueError("Each equation must be a sphere in x,y,z.")
    R2_1 = simplify(s1[0].subs({x:0, y:0}))
    R2_2 = simplify(s2[0].subs({x:0, y:0}))
    # sort so R2_big ≥ R2_small
    R2_big, R2_small = sorted([R2_1, R2_2], reverse=True)
    R_big   = sqrt(R2_big)
    R_small = sqrt(R2_small)

    # 2) x,y bounds come from the projection of the outer sphere:
    x_lo, x_hi = -R_big, R_big
    y_lo, y_hi = -sqrt(R2_big - x**2), sqrt(R2_big - x**2)

    # 3) For each fixed (x,y), z runs in two disjoint vertical segments:
    #    Segment 1: from bottom of big sphere down to bottom of small sphere
    z1_lo = -sqrt(R2_big   - x**2 - y**2)
    z1_hi = -sqrt(R2_small - x**2 - y**2)
    #    Segment 2: from top of small sphere up to top of big sphere
    z2_lo =  sqrt(R2_small - x**2 - y**2)
    z2_hi =  sqrt(R2_big   - x**2 - y**2)

    return (simplify(x_lo), simplify(x_hi),
            simplify(y_lo), simplify(y_hi),
            [(simplify(z1_lo), simplify(z1_hi)),
             (simplify(z2_lo), simplify(z2_hi))])

# register it alongside your other handlers:


# ─── Dispatcher ────────────────────────────────────────────────────────────────
handlers = {
    frozenset({'cylinder','paraboloid'})   : handle_cylinder_paraboloid,
    frozenset({'paraboloid','paraboloid'}) : handle_paraboloid_paraboloid,
    frozenset({'plane','plane'})           : handle_plane_plane,
    frozenset({'sphere','cone'})           : handle_sphere_cone,
    frozenset({'sphere','paraboloid'})     : handle_sphere_paraboloid,
    frozenset({'sphere','cylinder'})       : handle_sphere_cylinder,
    frozenset({'sphere','sphere'})         : handle_sphere_sphere,
    frozenset({'cylinder','cone'})         : handle_cylinder_cone,
    frozenset({'cylinder','cylinder'})     : handle_cylinder_cylinder,
    frozenset({'cone','paraboloid'})       : handle_cone_paraboloid,
    frozenset({'sphere','plane'})          : handle_sphere_plane,
    frozenset({'cylinder','plane'})        : handle_cylinder_plane,
    frozenset({'paraboloid','plane'})      : handle_paraboloid_plane
}


def plot_cylinder_paraboloid(eq1, eq2,
                             res_xy=60,
                             res_theta=72,
                             res_z=10):
    """
    Plot the closed solid region where
      cylinder:   a*x^2 + b*y^2 = C   (vertical)
      paraboloid: z = f(x,y)
    Opens an interactive 3D plot in a new browser tab.
    """
    x, y, z = sp.symbols('x y z', real=True)

    # 1) auto-swap so cyl_eq is cylinder, para_eq is paraboloid
    if region_type(eq1) == 'cylinder':
        cyl_eq, para_eq = eq1, eq2
    else:
        cyl_eq, para_eq = eq2, eq1

    # 2) extract cylinder params a,b,C
    Pc = sp.Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = Pc.coeff_monomial(x**2)
    b = Pc.coeff_monomial(y**2)
    C = -Pc.coeff_monomial(1)
    rx = float(math.sqrt(C/a))
    ry = float(math.sqrt(C/b))

    # 3) get symbolic z-bounds
    x_lo, x_hi, y_lo, y_hi, z_lo_expr, z_hi_expr = \
        handle_cylinder_paraboloid(cyl_eq, para_eq)

    # lambdify
    f_z_lo = sp.lambdify((x, y), z_lo_expr, 'numpy')
    f_z_hi = sp.lambdify((x, y), z_hi_expr, 'numpy')

    # 4) build XY grid
    xs = np.linspace(-rx, rx, res_xy)
    ys = np.linspace(-ry, ry, res_xy)
    X, Y = np.meshgrid(xs, ys)
    mask = (a*X**2 + b*Y**2) <= float(C) + 1e-9

    # 5) evaluate surfaces, handle constant vs array
    Zlo_raw = f_z_lo(X, Y)
    Zhi_raw = f_z_hi(X, Y)
    # broadcast constants to grid shape
    if np.ndim(Zlo_raw) == 0:
        Zlo = np.full_like(X, float(Zlo_raw))
    else:
        Zlo = np.array(Zlo_raw, dtype=float)
    if np.ndim(Zhi_raw) == 0:
        Zhi = np.full_like(X, float(Zhi_raw))
    else:
        Zhi = np.array(Zhi_raw, dtype=float)
    # mask outside
    Zlo[~mask] = np.nan
    Zhi[~mask] = np.nan

    # 6) side-wall extrusion along cylinder boundary
    thetas = np.linspace(0, 2*math.pi, res_theta)
    x_c = rx * np.cos(thetas)
    y_c = ry * np.sin(thetas)
    zlo_c = f_z_lo(x_c, y_c)
    zhi_c = f_z_hi(x_c, y_c)
    # broadcast constants if needed
    if np.ndim(zlo_c) == 0:
        zlo_c = np.full_like(x_c, float(zlo_c))
    else:
        zlo_c = np.array(zlo_c, dtype=float)
    if np.ndim(zhi_c) == 0:
        zhi_c = np.full_like(x_c, float(zhi_c))
    else:
        zhi_c = np.array(zhi_c, dtype=float)
    # interpolate in z
    t = np.linspace(0, 1, res_z)[:, None]
    Zc = (1 - t)*zhi_c[None, :] + t*zlo_c[None, :]
    Xc = np.tile(x_c, (res_z, 1))
    Yc = np.tile(y_c, (res_z, 1))

    # 7) build figure
    fig = go.Figure([
        go.Surface(x=X, y=Y, z=Zlo, showscale=False, opacity=0.8, name='Lower'),
        go.Surface(x=X, y=Y, z=Zhi, showscale=False, opacity=0.8, name='Upper'),
        go.Surface(x=Xc, y=Yc, z=Zc, showscale=False, opacity=0.7, name='Wall'),
    ])
    fig.update_layout(
        title="Solid Region: Cylinder ∧ Paraboloid",
        scene=dict(aspectmode='auto'),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # 8) render
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f'file://{tmp.name}')
def plot_cylinder_cone(eq1, eq2, resolution=60):
    """
    Plots the closed intersection of one vertical cylinder and one cone.
    - eq1, eq2: sympy.Eq objects, one must be a cylinder (no z² term),
      the other a cone (z² = Q(x,y)).
    - resolution: grid size per axis.
    Opens in a new browser tab.
    """
    # 1) Symbols
    x, y, z = sp.symbols('x y z', real=True)

    # 2) Decide which is cylinder vs cone
    def is_cylinder(eq):
        P = sp.Poly(eq.lhs - eq.rhs, x, y, z)
        return P.coeff_monomial(z**2) == 0 and (P.coeff_monomial(x**2) or P.coeff_monomial(y**2))
    def is_cone(eq):
        P = sp.Poly(eq.lhs - eq.rhs, x, y, z)
        return P.coeff_monomial(z**2) != 0 and P.coeff_monomial(1) == 0

    if   is_cylinder(eq1) and is_cone(eq2):
        cyl_eq, cone_eq = eq1, eq2
    elif is_cylinder(eq2) and is_cone(eq1):
        cyl_eq, cone_eq = eq2, eq1
    else:
        raise ValueError("Need one vertical cylinder (no z²) and one cone (z² = Q).")

    # 3) Cylinder: a x^2 + b y^2 = C_val
    Pc = sp.Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = float(Pc.coeff_monomial(x**2))
    b = float(Pc.coeff_monomial(y**2))
    C_val = -float(Pc.coeff_monomial(1))
    if a <= 0 or b <= 0 or C_val <= 0:
        raise ValueError("Invalid cylinder—must have a>0, b>0, C>0.")
    rx = math.sqrt(C_val / a)
    ry = math.sqrt(C_val / b)

    # 4) Cone: z^2 = Q_expr(x,y)
    sol = sp.solve(cone_eq, z**2)
    if not sol:
        raise ValueError("Cone must be solvable for z².")
    Q_expr = sol[0]
    fQ = sp.lambdify((x, y), Q_expr, 'numpy')

    # 5) Figure out maximum |z| over the circular footprint
    xs = np.linspace(-rx, rx, resolution)
    ys = np.linspace(-ry, ry, resolution)
    X2d, Y2d = np.meshgrid(xs, ys, indexing='xy')
    mask = (a*X2d**2 + b*Y2d**2) <= C_val
    Zvals = np.sqrt(np.clip(fQ(X2d, Y2d), 0, None))
    Zmax = np.nanmax(Zvals)

    # 6) Build the three surface meshes

    # 6a) Cylinder side (ellipse in XY, full Z-range)
    thetas = np.linspace(0, 2*np.pi, resolution)
    Zc   = np.linspace(-Zmax, Zmax, resolution)
    Th2, Z2 = np.meshgrid(thetas, Zc, indexing='xy')
    Xc = rx * np.cos(Th2)
    Yc = ry * np.sin(Th2)

    # 6b) Cone top and bottom (over XY grid, masked by cylinder)
    Z_top    =  np.sqrt(np.clip(fQ(X2d, Y2d), 0, None))
    Z_bottom = -Z_top
    Z_top[~mask]    = np.nan
    Z_bottom[~mask] = np.nan

    # 7) Plotly figure
    fig = go.Figure()

    # 7a) Cylinder side
    fig.add_trace(go.Surface(
        x=Xc, y=Yc, z=Z2,
        showscale=False,
        name='Cylinder side',
        colorscale='Viridis',
        opacity=0.75,
        lighting=dict(ambient=0.5, diffuse=0.6, roughness=0.9)
    ))

    # 7b) Cone “caps”
    fig.add_trace(go.Surface(
        x=X2d, y=Y2d, z=Z_top,
        showscale=False,
        name='Cone top',
        colorscale='Inferno',
        opacity=0.9,
        lighting=dict(ambient=0.6, diffuse=0.7, roughness=0.5)
    ))
    fig.add_trace(go.Surface(
        x=X2d, y=Y2d, z=Z_bottom,
        showscale=False,
        name='Cone bottom',
        colorscale='Inferno',
        opacity=0.9,
        lighting=dict(ambient=0.6, diffuse=0.7, roughness=0.5)
    ))

    # 8) Layout tweaks for beauty
    fig.update_layout(
        template='plotly_white',
        title="Bounded Intersection of Cylinder and Cone",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.8))
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    # 9) Write and open exactly as before
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f'file://{tmp.name}')
def plot_cone_paraboloid(eq1, eq2, res_r=80, res_theta=80):
    """
    Plot the solid intersection of a cone and a paraboloid:
      • Cone:        z^2 = A*(x^2 + y^2)
      • Paraboloid:  z   = a*(x^2 + y^2) + c
    Renders two surfaces (lower & upper) in Viridis, with 70% opacity,
    grey scene background, and opens in a browser tab.
    """
    # 1) Identify which is cone vs paraboloid
    x, y, z = symbols('x y z', real=True)
    t1, t2 = region_type(eq1), region_type(eq2)
    if t1 == 'cone':
        cone_eq, para_eq = eq1, eq2
    else:
        cone_eq, para_eq = eq2, eq1

    # 2) Extract cone coefficient A from z^2 = A*(x^2+y^2)
    sol_cone = solve(cone_eq, z**2)
    if not sol_cone:
        raise ValueError("Cannot isolate z^2 for the cone.")
    A = float(Poly(sol_cone[0], x, y).coeff_monomial(x**2))

    # 3) Extract paraboloid parameters a, c from z = a*(x^2+y^2)+c
    sol_para = solve(para_eq, z)
    if len(sol_para) != 1:
        raise ValueError("Paraboloid must solve to a single z=...")
    para_expr = sol_para[0]
    Pp = Poly(para_expr, x, y)
    a = float(Pp.coeff_monomial(x**2))
    c = float(Pp.coeff_monomial(1))

    # 4) Solve for intersection radius: (a*u + c)^2 = A*u , where u = r^2
    u = symbols('u', real=True)
    sol_u = solve((a*u + c)**2 - A*u, u)
    u_vals = [float(s) for s in sol_u if s.is_real and float(s) > 0]
    if not u_vals:
        raise ValueError("No positive real intersection radius.")
    u0 = max(u_vals)
    r_int = math.sqrt(u0)

    # 5) Build polar grid
    thetas = np.linspace(0, 2*math.pi, res_theta)
    rs = np.linspace(0, r_int, res_r)
    Rg, Tg = np.meshgrid(rs, thetas)  # shape (res_theta, res_r)
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)

    # 6) Compute upper/lower z surfaces
    Z_para = a * (Rg**2) + c
    Z_cone = np.sqrt(A) * Rg
    Z_lo = np.minimum(Z_para, Z_cone)
    Z_hi = np.maximum(Z_para, Z_cone)

    # 7) Create Plotly surfaces
    surf_lo = go.Surface(
        x=X, y=Y, z=Z_lo,
        showscale=False, opacity=0.7, colorscale='Viridis',
        name="Lower Boundary"
    )
    surf_hi = go.Surface(
        x=X, y=Y, z=Z_hi,
        showscale=False, opacity=0.7, colorscale='Viridis',
        name="Upper Boundary"
    )

    fig = go.Figure([surf_lo, surf_hi])
    fig.update_layout(
        title="Intersection: Cone ∧ Paraboloid",
        scene=dict(
            bgcolor="rgb(240,240,240)",
            aspectmode='cube',
            xaxis=dict(showbackground=True, backgroundcolor="rgb(200,200,200)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(200,200,200)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(200,200,200)"),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # 8) Render and open in new browser tab
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")

# Example usage:
# eq1 = sp.Eq(z**2, 2*(x**2 + y**2))
# eq2 = sp.Eq(z, 2*(x**2 + y**2))
# plot_cone_paraboloid(eq1, eq2)
def plot_paraboloid_cone(eq1, eq2,
                         res_r=80, res_theta=80, res_side_z=80):
    """
    Plots the solid intersection of a circular paraboloid
        z = a*(x^2 + y^2) + c
    and a circular cone
        z^2 = K*(x^2 + y^2).
    You’ll get one browser tab showing the full “bowl‐on‐cone” volume.
    """
    # 1) Identify which equation is which
    x, y, z = symbols('x y z', real=True)
    if region_type(eq1) == 'paraboloid':
        para_eq, cone_eq = eq1, eq2
    else:
        para_eq, cone_eq = eq2, eq1

    # 2) Extract paraboloid parameters: z = a r^2 + c
    sol_para = solve(para_eq, z)
    if len(sol_para) != 1:
        raise ValueError("Paraboloid must solve to z = f(x,y)")
    z_para = sol_para[0]
    Pp = Poly(z_para, x, y)
    a  = float(Pp.coeff_monomial(x**2))
    c  = float(Pp.coeff_monomial(1))
    if abs(a - float(Pp.coeff_monomial(y**2))) > 1e-6:
        raise ValueError("Paraboloid must be circular (same x²,y² coeff)")

    # 3) Extract cone parameter: z^2 = K r^2
    sol_k = solve(cone_eq, z**2)
    if not sol_k:
        raise ValueError("Cone must solve to z^2 = K*(x^2+y^2)")
    Pc = Poly(sol_k[0], x, y)
    K  = float(Pc.coeff_monomial(x**2))
    if abs(K - float(Pc.coeff_monomial(y**2))) > 1e-6:
        raise ValueError("Cone must be circular (same x²,y² coeff)")

    # 4) Find intersection radius r_p > 0 via (a r^2 + c)^2 = K r^2
    A = a*a
    B = 2*a*c - K
    C = c*c
    disc = B*B - 4*A*C
    if disc < 0:
        raise ValueError("No real intersection circle")
    u1 = (-B + math.sqrt(disc)) / (2*A)
    u2 = (-B - math.sqrt(disc)) / (2*A)
    rp2 = max(u1, u2)
    if rp2 <= 0:
        raise ValueError("Positive intersection radius not found")
    rp = math.sqrt(rp2)

    # 5) Build parametric meshes
    thetas = np.linspace(0, 2*np.pi, res_theta)
    rs      = np.linspace(0, rp,       res_r)
    Rg, Tg  = np.meshgrid(rs, thetas)

    # Paraboloid cap
    Xp = Rg * np.cos(Tg)
    Yp = Rg * np.sin(Tg)
    Zp = a*(Rg**2) + c

    # Both cone nappes
    Zc_up = +np.sqrt(K)*Rg
    Zc_dn = -np.sqrt(K)*Rg

    # Side wall at r=rp between the two surfaces
    Zs = np.linspace(Zc_dn.min(), Zp.max(), res_side_z)
    Ts, Zs_mat = np.meshgrid(thetas, Zs)
    Xs = rp * np.cos(Ts)
    Ys = rp * np.sin(Ts)

    # 6) Plot all pieces in one figure
    fig = go.Figure([
        go.Surface(x=Xp, y=Yp, z=Zp,     showscale=False, opacity=0.7, name="Paraboloid"),
        go.Surface(x=Xp, y=Yp, z=Zc_up,  showscale=False, opacity=0.7, name="Cone +z nappe"),
        go.Surface(x=Xp, y=Yp, z=Zc_dn,  showscale=False, opacity=0.7, name="Cone -z nappe"),
        go.Surface(x=Xs, y=Ys, z=Zs_mat, showscale=False, opacity=0.7, name="Side wall"),
    ])
    fig.update_layout(
        title="Intersection: Paraboloid ∧ Cone",
        scene=dict(aspectmode='auto')
    )

    # 7) Render in browser
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")
def plot_sphere_cylinder(eq1, eq2,
                         res_r=80,     # radial samples for caps
                         res_theta=80, # angular samples
                         res_side=80   # vertical samples on the cylinder
                        ):
    """
    Plot the 3D lens formed by intersecting:
      • Sphere:  x^2 + y^2 + z^2 = R^2
      • Cylinder: x^2 + y^2 = r0^2
    in one combined Plotly figure.
    """
    # 1) Identify which is sphere vs. cylinder
    x,y,z = symbols('x y z', real=True)
    if region_type(eq1)=='sphere':
        sph_eq, cyl_eq = eq1, eq2
    else:
        sph_eq, cyl_eq = eq2, eq1

    # 2) Extract sphere radius R
    Ps = Poly(sph_eq.lhs - sph_eq.rhs, x, y, z)
    c0_s = Ps.coeff_monomial(1)
    cx2 = Ps.coeff_monomial(x**2)
    R2  = -c0_s/cx2
    R   = math.sqrt(float(R2))

    # 3) Extract cylinder radius r0
    Pc   = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    c0_c = Pc.coeff_monomial(1)
    a    = Pc.coeff_monomial(x**2)
    # for a circular cylinder a==coeff of y^2
    r0  = math.sqrt(float(-c0_c/a))

    # 4) Compute z‐cap height where sphere meets cylinder
    if r0 > R:
        raise ValueError("Cylinder radius exceeds sphere radius → no intersection")
    zcap = math.sqrt(R*R - r0*r0)

    # 5) Build meshes

    # 5A) Top spherical cap: r∈[0,r0], θ∈[0,2π]
    rs     = np.linspace(0, r0, res_r)
    thetas = np.linspace(0, 2*np.pi, res_theta)
    Rg, Tg = np.meshgrid(rs, thetas)
    Xcap = Rg * np.cos(Tg)  
    Ycap = Rg * np.sin(Tg)
    Ztop =  np.sqrt(np.clip(R*R - Rg**2, 0, None))   # top cap
    Zbot = -np.sqrt(np.clip(R*R - Rg**2, 0, None))   # bottom cap

    # 5B) Cylinder side wall: r = r0, θ∈[0,2π], z∈[-zcap, +zcap]
    zs     = np.linspace(-zcap, zcap, res_side)
    Thet, Zs_mat = np.meshgrid(thetas, zs)
    Xside = r0 * np.cos(Thet)
    Yside = r0 * np.sin(Thet)

    # 6) Plot
    fig = go.Figure([
        # top cap
        go.Surface(x=Xcap, y=Ycap, z=Ztop,
                   showscale=False, opacity=0.8, name="Top Cap"),
        # bottom cap
        go.Surface(x=Xcap, y=Ycap, z=Zbot,
                   showscale=False, opacity=0.8, name="Bottom Cap"),
        # cylindrical wall
        go.Surface(x=Xside, y=Yside, z=Zs_mat,
                   showscale=False, opacity=0.8, name="Side Wall"),
    ])
    fig.update_layout(
        title="Intersection: Sphere ∧ Cylinder",
        scene=dict(aspectmode='auto'),
    )

    # 7) Render in browser
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")

def plot_paraboloid_paraboloid(eq1, eq2,
                               res_r=80,
                               res_theta=160,
                               res_side_z=120):
    """
    Plot the *bounded* solid region lying between TWO circular paraboloids,
        z = a1*(x^2 + y^2) + c1   and   z = a2*(x^2 + y^2) + c2.

    The routine:

      1. Calls handle_paraboloid_paraboloid → symbolic limits.
      2. Recovers the common intersection radius R from those limits.
      3. Builds a polar grid (r,θ) with r∈[0,R] and θ∈[0,2π].
      4. Evaluates both z–surfaces, chooses lower/upper per point.
      5. Adds a side wall at r = R so the mesh is watertight.
      6. Writes a temporary HTML and opens it in your default browser.

    Parameters
    ----------
    eq1, eq2 : sympy.Eq
        Each must be recognised as a *paraboloid* by region_type().
    res_r, res_theta : int
        Radial / angular sampling resolution for the caps.
    res_side_z : int
        Number of vertical samples along the side‐wall.

    Raises
    ------
    ValueError if the two paraboloids are not circular or do not form a
    closed finite lens-shaped region.
    """
    # --- 0) Symbols
    x, y, z = sp.symbols('x y z', real=True)

    # --- 1) Sanity check tags
    if region_type(eq1) != 'paraboloid' or region_type(eq2) != 'paraboloid':
        raise ValueError("Both equations must be recognised as paraboloids")

    # --- 2) Use existing handler to get tidy symbolic surfaces + R
    x_lo, x_hi, y_lo, y_hi, z_lo_expr, z_hi_expr = \
        handle_paraboloid_paraboloid(eq1, eq2)

    # Intersection radius R is simply |x_hi| (symbolic) → float
    R_sym = sp.simplify(x_hi)          # should be +sqrt(R²)
    R = float(sp.N(R_sym))

    # Lambdify z surfaces
    f_z_lo = sp.lambdify((x, y), z_lo_expr, 'numpy')
    f_z_hi = sp.lambdify((x, y), z_hi_expr, 'numpy')

    # --- 3) Polar grid in the disc r ≤ R
    thetas = np.linspace(0, 2*np.pi, res_theta)
    rs     = np.linspace(0, R,       res_r)
    Rg, Tg = np.meshgrid(rs, thetas)             # (θ, r)
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)

    # --- 4) Evaluate the two z surfaces on that grid
    Z_lo = f_z_lo(X, Y)
    Z_hi = f_z_hi(X, Y)

    # Convert possible scalars to arrays
    if np.ndim(Z_lo) == 0:
        Z_lo = np.full_like(X, float(Z_lo))
    if np.ndim(Z_hi) == 0:
        Z_hi = np.full_like(X, float(Z_hi))

    # Ensure “lo” really is the lower branch everywhere
    idx = Z_lo > Z_hi
    Z_lo[idx], Z_hi[idx] = Z_hi[idx], Z_lo[idx]

    # --- 5) Side wall at r = R between Z_lo and Z_hi
    Z_side = np.linspace(Z_lo.min(), Z_hi.max(), res_side_z)
    Theta_side, Z_side_mat = np.meshgrid(thetas, Z_side)
    X_side = R * np.cos(Theta_side)
    Y_side = R * np.sin(Theta_side)

    # --- 6) Build Plotly figure
    fig = go.Figure([
        go.Surface(x=X,      y=Y,      z=Z_lo,
                   showscale=False, opacity=0.7,
                   colorscale='Viridis', name="Lower paraboloid"),
        go.Surface(x=X,      y=Y,      z=Z_hi,
                   showscale=False, opacity=0.7,
                   colorscale='Viridis', name="Upper paraboloid"),
        go.Surface(x=X_side, y=Y_side, z=Z_side_mat,
                   showscale=False, opacity=0.7,
                   colorscale='Greys',   name="Side wall")
    ])
    fig.update_layout(
        title="Intersection: Paraboloid ∧ Paraboloid",
        scene=dict(aspectmode='auto'),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # --- 7) Render → temp HTML
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")
def plot_sphere_cone(eq1, eq2,
                     res_r=80,
                     res_theta=160):
    """
    Plot the *bounded* region above the cone z^2 = K*(x^2+y^2)
    and below the sphere x^2+y^2+z^2 = R^2.
    Opens a single HTML (Plotly) in your browser.
    """
    x,y,z = symbols('x y z', real=True)

    # 1) Figure out which is which
    #    sphere: all x^2,y^2,z^2 coeffs equal & no linear/cross terms
    def _is_sphere(eq):
        P = Poly(eq.lhs - eq.rhs, x,y,z)
        c2 = P.coeff_monomial(x**2)
        return (P.total_degree()==2
                and c2!=0
                and P.coeff_monomial(y**2)==c2
                and P.coeff_monomial(z**2)==c2
                and all(P.coeff_monomial(m)==0
                        for m in [(1,0,0),(0,1,0),(0,0,1),
                                  (1,1,0),(1,0,1),(0,1,1)]))
    if _is_sphere(eq1):
        sph_eq, cone_eq = eq1, eq2
    else:
        sph_eq, cone_eq = eq2, eq1

    # 2) Extract R from sphere: c*(x²+y²+z²) + c0 = 0 → R² = -c0/c
    Ps = Poly(sph_eq.lhs - sph_eq.rhs, x,y,z)
    c_x2 = Ps.coeff_monomial(x**2)
    c0   = Ps.coeff_monomial(1)
    R2   = float(-c0 / c_x2)
    R    = math.sqrt(R2)

    # 3) Extract K from cone: z² = K*(x²+y²)
    sol = solve(cone_eq, z**2)
    if not sol:
        raise ValueError("Could not isolate z² in cone equation.")
    Q = sol[0]
    Pc = Poly(Q, x,y)
    K  = float(Pc.coeff_monomial(x**2))
    # (we assume circularity so coeff of y² matches)
    if abs(K - float(Pc.coeff_monomial(y**2)))>1e-6:
        raise ValueError("Cone must be circular: z² = K(x²+y²)")

    # 4) Intersection circle radius
    r_int = R / math.sqrt(1 + K)

    # 5) Build a (r,θ) grid over 0 ≤ r ≤ r_int
    thetas = np.linspace(0, 2*math.pi, res_theta)
    rs     = np.linspace(0, r_int,    res_r)
    Rg, Tg = np.meshgrid(rs, thetas)

    # 6) Convert to (x,y) and compute both z-surfaces
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    Z_cone   = np.sqrt(K) * Rg
    Z_sphere = np.sqrt(np.clip(R2 - Rg**2, 0, None))

    # 7) Plot the two surfaces
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_cone,
        showscale=False,
        opacity=0.6,
        name="Cone"
    ))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_sphere,
        showscale=False,
        opacity=0.6,
        name="Sphere cap"
    ))

    fig.update_layout(
        title="Bounded intersection: sphere ∧ cone",
        scene=dict(
            xaxis=dict(title="x", range=[-R, R]),
            yaxis=dict(title="y", range=[-R, R]),
            zaxis=dict(title="z", range=[0, R]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 8) Write out and open in a browser tab
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")
def plot_sphere_plane(eq1, eq2, res_sphere=60, res_plane=60):
    """
    Plot an origin-centered sphere x^2+y^2+z^2=R^2 and the plane A x+B y+C z=D,
    clipping the plane to its elliptical intersection with the sphere.
    Opens a new browser tab with the interactive 3D figure.
    """
    # --- 1) Identify sphere vs plane ---
    x,y,z = symbols('x y z', real=True)
    def is_sphere(e):
        P = Poly(e.lhs - e.rhs, x,y,z)
        c2 = P.coeff_monomial(x**2)
        return (P.total_degree()==2 and c2!=0
                and P.coeff_monomial(y**2)==c2
                and P.coeff_monomial(z**2)==c2
                and all(P.coeff_monomial(m)==0
                        for m in [(1,0,0),(0,1,0),(0,0,1),
                                  (1,1,0),(1,0,1),(0,1,1)]))
    if is_sphere(eq1):
        sph_eq, plane_eq = eq1, eq2
    else:
        sph_eq, plane_eq = eq2, eq1

    # --- 2) Extract R² from sphere eq ---
    sol_z2 = solve(sph_eq, z**2)
    if not sol_z2:
        raise ValueError("Could not isolate z² in sphere.")
    R2 = simplify(sol_z2[0].subs({x:0,y:0}))
    R  = float(sp.sqrt(R2))

    # --- 3) Extract plane coefficients A,B,C,D ---
    expr = plane_eq.lhs - plane_eq.rhs
    Pp = Poly(expr, x,y,z)
    A = float(Pp.coeff_monomial(x))
    B = float(Pp.coeff_monomial(y))
    C = float(Pp.coeff_monomial(z))
    D = -float(Pp.coeff_monomial(1))

    # --- 4) Sphere surface mesh in spherical coords ---
    u = np.linspace(0, np.pi, res_sphere)
    v = np.linspace(0, 2*np.pi, res_sphere)
    U, V = np.meshgrid(u, v)
    Xs = R * np.sin(U) * np.cos(V)
    Ys = R * np.sin(U) * np.sin(V)
    Zs = R * np.cos(U)

    # --- 5) Plane mesh clipped to sphere intersection ---
    # Parametric grid in x-y over bounding box [-R,R]^2
    x_lin = np.linspace(-R, R, res_plane)
    y_lin = np.linspace(-R, R, res_plane)
    Xp, Yp = np.meshgrid(x_lin, y_lin)
    # Compute z from plane; when C=0 this would blow up, but C=0 -> vertical plane
    if abs(C) < 1e-8:
        # For vertical planes, solve for x or y instead:
        # Here we handle only non-vertical for simplicity
        raise ValueError("Vertical planes (C=0) not supported in this plotting routine.")
    Zp = (D - A*Xp - B*Yp)/C
    # Mask to keep only points on plane inside sphere
    mask = (Xp**2 + Yp**2 + Zp**2) <= R2
    Xp[~mask] = np.nan
    Yp[~mask] = np.nan
    Zp[~mask] = np.nan

    # --- 6) Build Plotly figure ---
    fig = go.Figure()

    # Sphere
    fig.add_trace(go.Surface(
        x=Xs, y=Ys, z=Zs,
        opacity=0.6, showscale=False,
        colorscale='Viridis',
        name='Sphere'))

    # Clipped plane
    fig.add_trace(go.Surface(
        x=Xp, y=Yp, z=Zp,
        opacity=0.6, showscale=False,
        colorscale=[[0, 'lightpink'], [1, 'lightpink']],
        name='Plane ∩ Sphere'))

    fig.update_layout(
        title="Sphere ∧ Plane",
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # --- 7) Render to a temp HTML and open in browser ---
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name, auto_open=False)
    webbrowser.open_new_tab(f"file://{tmp.name}")
def plot_sphere_paraboloid(eq1, eq2,
                           res_r=80,
                           res_theta=80):
    """
    Plot the bounded region between
      • Sphere:     x^2 + y^2 + z^2 = R^2
      • Paraboloid: z = a*(x^2 + y^2) + c

    If they do NOT form a closed finite lens, we display the
    would-be limits and then bail out with a “cannot compute volume”
    message instead of plotting.
    """
    x,y,z,u = symbols('x y z u', real=True)

    # 1) Identify sphere vs paraboloid
    from sympy import Poly
    def _is_sphere(e):
        P = Poly(e.lhs - e.rhs, x,y,z)
        c2 = P.coeff_monomial(x**2)
        return (P.total_degree()==2
                and c2!=0
                and P.coeff_monomial(y**2)==c2
                and P.coeff_monomial(z**2)==c2
                and all(P.coeff_monomial(m)==0
                        for m in [(1,0,0),(0,1,0),(0,0,1),
                                  (1,1,0),(1,0,1),(0,1,1)]))
    if _is_sphere(eq1):
        sph_eq, para_eq = eq1, eq2
    else:
        sph_eq, para_eq = eq2, eq1

    # 2) Extract R² from sphere
    Ps = Poly(sph_eq.lhs - sph_eq.rhs, x,y,z)
    c0 = Ps.coeff_monomial(1)
    cx2 = Ps.coeff_monomial(x**2)
    R2 = float(-c0 / cx2)
    R  = math.sqrt(R2)

    # 3) Extract paraboloid parameters z = a*r^2 + c
    sols = solve(para_eq, z)
    if len(sols)!=1:
        raise ValueError("Paraboloid must solve to z = f(x,y)")
    z_para = sols[0]
    Pp = Poly(z_para, x,y)
    a = float(Pp.coeff_monomial(x**2))
    c = float(Pp.coeff_monomial(1))
    if abs(a - float(Pp.coeff_monomial(y**2)))>1e-6:
        raise ValueError("Paraboloid must be circular (same x², y² coeff)")

    # 4) Solve for u = r² from intersection:  (a u + c)² = R² - u
    eq_u = (a*u + c)**2 - (R2 - u)
    roots = solve(eq_u, u)
    # keep only real ≥0
    u_vals = [float(r.evalf()) for r in roots
              if r.is_real and float(r)>=0]
    if not u_vals:
        # no finite intersection → unbounded or empty
        # Produce the symbolic limits we *would* have printed
        print("Would-be limits (no bounded region):")
        print(f"  x from -sqrt(R2/c_x) to sqrt(R2/c_x)  # R2={R2}, c_x={cx2}")
        print("  y from -sqrt(... ) to sqrt(... )")
        print("  z from a*(x^2+y^2)+c to ±sqrt(R² - x^2 - y^2)")
        print("\nCannot compute volume: region is not a closed, bounded lens.")
        return

    # pick the largest positive root
    u_int = max(u_vals)
    r_int = math.sqrt(u_int)

    # 5) Build polar mesh 0 ≤ r ≤ r_int
    thetas = np.linspace(0, 2*math.pi, res_theta)
    rs     = np.linspace(0, r_int,    res_r)
    Rg, Tg = np.meshgrid(rs, thetas)

    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    Zp = a*(Rg**2) + c               # paraboloid
    Zs = np.sqrt(np.clip(R2 - Rg**2, 0, None))  # sphere cap

    # 6) Plot only the two bounding surfaces
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zp,
        showscale=False, opacity=0.7, name="Paraboloid"
    ))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zs,
        showscale=False, opacity=0.7, name="Sphere cap"
    ))
    fig.update_layout(
        title="Sphere ∧ Paraboloid (bounded lens)",
        scene=dict(
            xaxis=dict(range=[-R, R]),
            yaxis=dict(range=[-R, R]),
            zaxis=dict(range=[min(Zp.min(),0), R]),
            aspectmode='cube'
        ),
        margin=dict(l=0,r=0,b=0,t=30)
    )

    # 7) Render in browser
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")
def plot_sphere_sphere(eq1, eq2, res_phi=80, res_theta=80):
    """
    Plots the region between two concentric spheres:
      eq1: x^2+y^2+z^2 = R1^2
      eq2: x^2+y^2+z^2 = R2^2
    R1 and R2 may be in any order; the region is the shell between them.
    """
    # Identify both as spheres and extract radii
    x, y, z = symbols('x y z', real=True)
    Ps1 = Poly(eq1.lhs - eq1.rhs, x, y, z)
    Ps2 = Poly(eq2.lhs - eq2.rhs, x, y, z)
    R21 = float(-Ps1.coeff_monomial(1) / Ps1.coeff_monomial(x**2))
    R22 = float(-Ps2.coeff_monomial(1) / Ps2.coeff_monomial(x**2))
    R1, R2 = math.sqrt(R21), math.sqrt(R22)
    R_small, R_big = sorted([R1, R2])

    # Parameter grids
    phi   = np.linspace(0, np.pi,   res_phi)
    theta = np.linspace(0, 2*np.pi, res_theta)
    Phi, Theta = np.meshgrid(phi, theta)

    # Outer sphere surface
    Xb = R_big * np.sin(Phi) * np.cos(Theta)
    Yb = R_big * np.sin(Phi) * np.sin(Theta)
    Zb = R_big * np.cos(Phi)

    # Inner sphere surface
    Xs = R_small * np.sin(Phi) * np.cos(Theta)
    Ys = R_small * np.sin(Phi) * np.sin(Theta)
    Zs = R_small * np.cos(Phi)

    # Plot
    fig = go.Figure([
        go.Surface(x=Xb, y=Yb, z=Zb,
                   showscale=False, opacity=0.6, name="Outer Sphere"),
        go.Surface(x=Xs, y=Ys, z=Zs,
                   showscale=False, opacity=0.6, name="Inner Sphere")
    ])
    fig.update_layout(
        title="Intersection Region: Shell between Sphere ∧ Sphere",
        scene=dict(aspectmode='auto')
    )

    # Render in browser
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")
def plot_paraboloid_plane(eq1, eq2, resolution=60):
    """
    Plot the bounded region defined by the intersection of a paraboloid and a plane.
    - eq1, eq2: sympy Eq objects, one paraboloid (z = f(x,y)), one plane (z = g(x,y)).
    - resolution: number of points per axis for the meshgrid.
    Opens an interactive 3D plot in a new browser tab.
    """
    # 1) Identify which is paraboloid vs plane
    if region_type(eq1) == 'paraboloid':
        para_eq, plane_eq = eq1, eq2
    else:
        para_eq, plane_eq = eq2, eq1

    # 2) Get integration limits
    x_lo, x_hi, y_lo_expr, y_hi_expr, z_lo_expr, z_hi_expr = handle_paraboloid_plane(para_eq, plane_eq)

    # 3) Sample x range and compute y-bounds safely
    xs = np.linspace(float(x_lo), float(x_hi), resolution)
    y_values = []
    for xi in xs:
        try:
            ylo_val = y_lo_expr.subs(x, xi)
            yhi_val = y_hi_expr.subs(x, xi)
            ylo = complex(ylo_val)
            yhi = complex(yhi_val)
            if abs(ylo.imag) < 1e-6: y_values.append(ylo.real)
            if abs(yhi.imag) < 1e-6: y_values.append(yhi.real)
        except Exception:
            continue
    if not y_values:
        raise RuntimeError("Could not find any real y-bounds across x-samples.")
    y_min, y_max = min(y_values), max(y_values)
    ys = np.linspace(y_min, y_max, resolution)

    # 4) Build mesh and lambdify z
    X, Y = np.meshgrid(xs, ys)
    f_lo = lambdify((x, y), z_lo_expr, 'numpy')
    f_hi = lambdify((x, y), z_hi_expr, 'numpy')
    Zlo = f_lo(X, Y)
    Zhi = f_hi(X, Y)

    # 4a) if scalar result, broadcast to mesh shape
    if np.isscalar(Zlo):
        Zlo = np.full_like(X, float(Zlo))
    if np.isscalar(Zhi):
        Zhi = np.full_like(X, float(Zhi))

    # 5) Mask invalid (complex/nan) regions
    valid = np.isfinite(Zlo) & np.isfinite(Zhi) & (Zhi >= Zlo)
    Zlo_masked = np.where(valid, Zlo, np.nan)
    Zhi_masked = np.where(valid, Zhi, np.nan)

    # 6) Plot surfaces
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zlo_masked,
        showscale=False, opacity=0.7, name="Plane (lower)"
    ))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zhi_masked,
        showscale=False, opacity=0.7, name="Paraboloid (upper)"
    ))
    fig.update_layout(
        title="Intersection: Paraboloid ∧ Plane",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode='auto'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # 7) Open in browser
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")



def plot_cylinder_cylinder(eq1, eq2, resolution=50):
    """
    Plot the closed solid region where two cylinders intersect:
      • Vertical cylinder:    a*x^2 + b*y^2 = C
      • Perpendicular cylinder: involves z^2 + one of x^2 or y^2 = C2
    Uses a zero‐isosurface F(x,y,z)=max(F1,F2)=0 for a closed mesh.
    Opens in a new browser tab.
    """
    x, y, z = sp.symbols('x y z', real=True)

    # 1) Identify vertical vs perpendicular cylinder
    P1 = sp.Poly(eq1.lhs - eq1.rhs, x, y, z)
    P2 = sp.Poly(eq2.lhs - eq2.rhs, x, y, z)
    if P1.coeff_monomial(z**2) == 0:
        Pv, Pp = P1, P2
    else:
        Pv, Pp = P2, P1

    # 2) Extract vertical cylinder params: a x^2 + b y^2 = C
    a = Pv.coeff_monomial(x**2)
    b = Pv.coeff_monomial(y**2)
    C = -Pv.coeff_monomial(1)
    Rx = math.sqrt(float(C/a))
    Ry = math.sqrt(float(C/b))

    # 3) Extract perpendicular cylinder: solve z^2 = (C2 - α·var^2)/β
    C2 = -Pp.coeff_monomial(1)
    p_x2 = Pp.coeff_monomial(x**2)
    p_y2 = Pp.coeff_monomial(y**2)
    p_z2 = Pp.coeff_monomial(z**2)
    if p_y2 and p_z2:
        # axis ∥ x-axis: y²+z²=C2
        Q_expr = (C2 - p_y2*y**2) / p_z2
    else:
        # axis ∥ y-axis: x²+z²=C2
        Q_expr = (C2 - p_x2*x**2) / p_z2

    # 4) lambdify implicit functions
    fQ = sp.lambdify((x, y), Q_expr, 'numpy')

    # 5) Build sampling grid
    xs = np.linspace(-Rx, Rx, resolution)
    ys = np.linspace(-Ry, Ry, resolution)
    # limit z by maximum possible from Q_expr at center
    zmax = math.sqrt(float(sp.simplify(Q_expr.subs({x:0,y:0}))))
    zs = np.linspace(-zmax, zmax, resolution)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    # 6) Evaluate F1 and F2, then F = max(F1,F2)
    F1 = a*X**2 + b*Y**2 - float(C)
    F2 = Z**2 - fQ(X, Y)
    # cast to numpy float
    F1 = np.array(F1, dtype=float)
    F2 = np.array(F2, dtype=float)
    V  = np.maximum(F1, F2)

    # 7) Plot the zero-isosurface F=0
    fig = go.Figure(data=go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=V.ravel(),
        isomin=0, isomax=0,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    fig.update_layout(
        title="Intersection of Two Cylinders",
        scene=dict(aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # 8) Render and open
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f"file://{tmp.name}")
def plot_plane_plane(eq1, eq2, resolution=60):
    """
    Plot the bounded tetrahedron cut out by two planes eq1, eq2
    and the coordinate planes x=0, y=0, z=0.
    Opens a browser tab with the 3D plot.
    """

    # 1) Solve each for z = f(x,y)
    x,y,z = sp.symbols('x y z', real=True)
    f1 = sp.solve(eq1, z)[0]
    f2 = sp.solve(eq2, z)[0]

    # 2) Determine which is lower/higher by sampling at (0,0)
    f_lo_sym, f_hi_sym = (f1, f2) if float(f1.subs({x:0, y:0})) < float(f2.subs({x:0, y:0})) else (f2, f1)

    # 3) Project intersection onto z=0: each plane gives A*x+B*y = C
    def plane_coeffs(eq):
        expr = sp.simplify(eq.lhs - eq.rhs).subs(z, 0)
        A = float(expr.coeff(x))
        B = float(expr.coeff(y))
        C = float(-expr.subs({x:0, y:0}))
        return A,B,C

    A1,B1,C1 = plane_coeffs(eq1)
    A2,B2,C2 = plane_coeffs(eq2)

    # The true base region is the intersection of the two half-planes plus x>=0,y>=0
    # Compute numeric x/y ranges by finding where those lines cross axes:
    # We'll just take the convex polygon of x>=0,y>=0,A1 x+B1 y<=C1,A2 x+B2 y<=C2
    # and then sample inside it.
    # But for a tetrahedron, typically one plane is slanted above the other, so one dominates.
    # We’ll assume A1x+B1y<=C1 and A2x+B2y<=C2 both must hold.

    # Build a fine grid in the rectangle that bounds both polygons
    # First find naive axis intercepts:
    def intercepts(A,B,C):
        xs = []
        ys = []
        if abs(A)>1e-8:
            xs.append(C/A)
        if abs(B)>1e-8:
            ys.append(C/B)
        return xs, ys

    xs1, ys1 = intercepts(A1,B1,C1)
    xs2, ys2 = intercepts(A2,B2,C2)
    x_max = max(xs1+xs2+[0.0])
    y_max = max(ys1+ys2+[0.0])

    xs = np.linspace(0.0, x_max, resolution)
    ys = np.linspace(0.0, y_max, resolution)
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # Mask to the polygon where both planes are above z=0
    mask1 = (A1*X + B1*Y <= C1 + 1e-8)
    mask2 = (A2*X + B2*Y <= C2 + 1e-8)
    mask0 = (X>=-1e-8)&(Y>=-1e-8)
    base_mask = mask1 & mask2 & mask0

    # 4) Lambdify the symbolic bounds
    f_lo = sp.lambdify((x,y), f_lo_sym, 'numpy')
    f_hi = sp.lambdify((x,y), f_hi_sym, 'numpy')

    Zlo = f_lo(X, Y)
    Zhi = f_hi(X, Y)
    Z0  = np.zeros_like(Zlo)

    # Apply mask
    Zlo_masked = np.where(base_mask, Zlo, np.nan)
    Zhi_masked = np.where(base_mask, Zhi, np.nan)

    # 5) Plot
    fig = go.Figure()

    # bottom face z=0
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z0,
        showscale=False, opacity=0.4,
        colorscale=[[0.0, 'lightgray']],
        name='z=0'
    ))

    # lower slanted face
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zlo_masked,
        showscale=False, opacity=0.7,
        name=str(sp.simplify(f_lo_sym))
    ))
    # upper slanted face
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Zhi_masked,
        showscale=False, opacity=0.7,
        name=str(sp.simplify(f_hi_sym))
    ))

    # side wall x=0
    ys_wall = ys
    xs0 = np.zeros_like(ys_wall)
    zlo0 = f_lo(xs0, ys_wall)
    zhi0 = f_hi(xs0, ys_wall)
    fig.add_trace(go.Surface(
        x=np.vstack([xs0, xs0]),
        y=np.vstack([ys_wall, ys_wall]),
        z=np.vstack([zlo0, zhi0]),
        showscale=False, opacity=0.5,
        name='x=0'
    ))

    # side wall y=0
    xs_wall = xs
    ys0 = np.zeros_like(xs_wall)
    zlo1 = f_lo(xs_wall, ys0)
    zhi1 = f_hi(xs_wall, ys0)
    fig.add_trace(go.Surface(
        x=np.vstack([xs_wall, xs_wall]),
        y=np.vstack([ys0, ys0]),
        z=np.vstack([zlo1, zhi1]),
        showscale=False, opacity=0.5,
        name='y=0'
    ))

    # side wall A1 x + B1 y = C1 (clip to the base_mask boundary)
    # parametrize t from 0→1 → (x(t),y(t)) traces along intersection line
    t = np.linspace(0,1,resolution)
    xb = t*(C1/A1)        # when B1≠0, better to param via y=0→C1/B1 but ok for generic
    yb = (C1 - A1*xb)/B1
    zlo_b = f_lo(xb, yb)
    zhi_b = f_hi(xb, yb)
    fig.add_trace(go.Surface(
        x=np.vstack([xb, xb]),
        y=np.vstack([yb, yb]),
        z=np.vstack([zlo_b, zhi_b]),
        showscale=False, opacity=0.5,
        name=f'{A1}x+{B1}y={C1}'
    ))

    fig.update_layout(
        title="Intersection of Two Planes",
        scene=dict(aspectmode='auto'),
        margin=dict(l=0,r=0,b=0,t=30)
    )

    # 6) Render
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    fig.write_html(tmp.name)
    webbrowser.open_new_tab(f'file://{tmp.name}')
from sympy import integrate, simplify, symbols
from sympy import Eq
from sympy import integrate, simplify, symbols, Eq
from sympy.core.sympify import SympifyError

def find_z_bounds(surfs):
    if len(surfs) == 1:
        sols = solve(surfs[0], z)
        # sort by f(0,0)
        sols = sorted(sols, key=lambda f: float(f.subs({x:0,y:0}).evalf()))
        return simplify(sols[0]), simplify(sols[1])
    elif len(surfs) == 2:
        s0 = solve(surfs[0], z)[0]
        s1 = solve(surfs[1], z)[0]
        sols = sorted([s0, s1], key=lambda f: float(f.subs({x:0,y:0}).evalf()))
        return simplify(sols[0]), simplify(sols[1])
    else:
        raise ValueError("Must supply 1 or 2 z-surfaces.")
    

def find_xy_bounds(surfs, cons, z_bounds):
    """
    Returns (y_lo, y_hi), (x_lo, x_hi):
      • If len(cons)==1 & len(surfs)==1: one z-surface + one xy-constraint.
      • If len(cons)==0 & len(surfs)==2: intersection of two z-surfaces.

    surfs: list of Eq(z, f(x,y)), length 1 or 2
    cons:  list of Eq in x,y (e.g. cylinder or vertical plane), length 0 or 1
    z_bounds: tuple (z_lo, z_hi)
    """
    x, y, z = symbols('x y z', real=True)
    z_lo, z_hi = z_bounds

    # ── Case A: 1 side-constraint + 1 z-surface ───────────────────────────
    if len(cons)==1 and len(surfs)==1:
        c = cons[0]
        # y-bounds from c(x,y)=0
        y_sols = solve(c, y)
        y_vals = [float(sol.subs(x,0)) for sol in y_sols]
        y_lo = y_sols[y_vals.index(min(y_vals))]
        y_hi = y_sols[y_vals.index(max(y_vals))]
        # x-bounds from c(x,y=0)=0
        x_sols = solve(c.subs(y,0), x)
        x_vals = [float(sol) for sol in x_sols]
        x_lo = x_sols[x_vals.index(min(x_vals))]
        x_hi = x_sols[x_vals.index(max(x_vals))]
        return (y_lo, y_hi), (x_lo, x_hi)

    # ── Case B: intersection of two z-surfaces ───────────────────────────
    if len(cons)==0 and len(surfs)==2:
        # Δ(x,y) = z_hi - z_lo
        Δ = simplify(z_hi - z_lo)

        # 1) Solve Δ=0 for the two y-branches
        y_branches = solve(Δ, y)
        if len(y_branches) != 2:
            raise ValueError("Expected two y-branches from Δ=0")
        y1, y2 = [simplify(sol) for sol in y_branches]

        # 2) Discriminant D(x) of Δ viewed as quadratic in y: A y² + B y + C = 0
        P = Poly(Δ, y, x)
        A = P.coeff_monomial(y**2)
        B = P.coeff_monomial(y**1)
        C = P.coeff_monomial(1)
        D = simplify(B**2 - 4*A*C)

        # 3) Solve D(x) ≥ 0 to get projection onto x
        x_region = solve_univariate_inequality(D >= 0, x)
        # Accept Interval or Union → take overall hull
        if isinstance(x_region, Interval):
            x_lo, x_hi = x_region.start, x_region.end
        elif isinstance(x_region, Union):
            ivs = [iv for iv in x_region.args if isinstance(iv, Interval)]
            starts = [iv.start for iv in ivs]
            ends   = [iv.end   for iv in ivs]
            x_lo, x_hi = min(starts), max(ends)
        else:
            raise ValueError("Projection onto x not a single continuous interval")

        # 4) Order the two y-branches by sampling at midpoint
        x_mid = (x_lo + x_hi)/2
        if float(y1.subs(x, x_mid)) < float(y2.subs(x, x_mid)):
            y_lo, y_hi = y1, y2
        else:
            y_lo, y_hi = y2, y1

        return (simplify(y_lo), simplify(y_hi)), \
               (simplify(x_lo), simplify(x_hi))

    raise ValueError(
        "find_xy_bounds only supports:\n"
        " • 1 side-constraint + 1 z-surface, or\n"
        " • 2 z-surfaces & no side-constraint"
    )


import numpy as np
import sympy as sp
from scipy import integrate

def safe_lambdify(sym_expr, args):
    """
    Wrap sympy_expr → numpy function, but replace sqrt(u) with sqrt(max(u,0))
    so you never take sqrt of a negative number.
    """
    # clamp_sqrt will zero-out negative inputs
    def clamp_sqrt(u):
        u = np.array(u)
        return np.sqrt(np.clip(u, 0.0, None))

    # Build a custom module dict: override 'sqrt' before falling back to numpy
    modules = [
        {
            'sqrt': clamp_sqrt,
            # you can also override other functions if needed:
            'abs':   np.abs,
            'log':   np.log,
            'sin':   np.sin,
            'cos':   np.cos,
            'tan':   np.tan,
            'exp':   np.exp
        },
        'numpy'    # then use standard numpy for everything else
    ]

    f = sp.lambdify(args, sym_expr, modules=modules)
    def f_safe(*nums):
        with np.errstate(invalid='ignore', divide='ignore'):
            out = f(*nums)
        # force real and finite
        out = np.real(out)
        return np.where(np.isfinite(out), out, np.nan)
    return f_safe

def compute_volume(eq1, eq2, method='scipy', mc_samples=200_000):
    """
    Robust volume between eq1 & eq2 using:
      - method='scipy': nested dblquad only
      - method='montecarlo': uniform sampling
    """
    # dispatch to your existing region_type/handlers
    types = frozenset({region_type(eq1), region_type(eq2)})
    if types not in handlers:
        raise ValueError(f"No handler for shapes {types}")
    handler = handlers[types]

    # get symbolic bounds
    x_lo_s, x_hi_s, y_lo_s, y_hi_s, z_lo_s, z_hi_s = handler(eq1, eq2)
    x, y = sp.symbols('x y', real=True)
    x_lo, x_hi = float(x_lo_s), float(x_hi_s)

    # lambdify *all* bounds via the new safe_lambdify
    f_y_lo = safe_lambdify(y_lo_s, (x,))
    f_y_hi = safe_lambdify(y_hi_s, (x,))
    f_z_lo = safe_lambdify(z_lo_s, (x, y))
    f_z_hi = safe_lambdify(z_hi_s, (x, y))

    if method == 'scipy':
        def integrand(y_val, x_val):
            dz = f_z_hi(x_val, y_val) - f_z_lo(x_val, y_val)
            # clamp negative thickness to zero
            return np.where((dz>0)&np.isfinite(dz), dz, 0.0)

        V, err = integrate.dblquad(
            integrand,
            x_lo, x_hi,
            lambda xv: f_y_lo(xv),
            lambda xv: f_y_hi(xv),
            epsabs=1e-8, epsrel=1e-8
        )
        return V

    elif method == 'montecarlo':
        # (a) find valid y-range
        xs = np.linspace(x_lo, x_hi, 500)
        ys_lo = f_y_lo(xs); ys_hi = f_y_hi(xs)
        valid = np.isfinite(ys_lo)&np.isfinite(ys_hi)
        if not np.any(valid):
            raise ValueError("No finite y-bounds")
        y_lo, y_hi = ys_lo[valid].min(), ys_hi[valid].max()

        # (b) find valid z-range on a grid
        xg = np.linspace(x_lo, x_hi, 100)
        yg = np.linspace(y_lo, y_hi, 100)
        Xg, Yg = np.meshgrid(xg, yg, indexing='xy')
        Zlo = f_z_lo(Xg, Yg); Zhi = f_z_hi(Xg, Yg)
        m2 = np.isfinite(Zlo)&np.isfinite(Zhi)
        if not np.any(m2):
            raise ValueError("No finite z-bounds")
        z_lo, z_hi = Zlo[m2].min(), Zhi[m2].max()

        # (c) random sampling
        X = np.random.uniform(x_lo, x_hi, mc_samples)
        Y = np.random.uniform(y_lo, y_hi, mc_samples)
        Z = np.random.uniform(z_lo, z_hi, mc_samples)
        inside = (
            (Y >= f_y_lo(X)) & (Y <= f_y_hi(X)) &
            (Z >= f_z_lo(X, Y)) & (Z <= f_z_hi(X, Y))
        )
        vol_box = (x_hi-x_lo)*(y_hi-y_lo)*(z_hi-z_lo)
        return inside.mean() * vol_box

    else:
        raise ValueError(f"Unknown method {method!r}")
# (A) Tag surfaces by re-using your region_type:
def tag_surface(eq_text):
    eq = parse_equation(eq_text)     # your existing parser
    return region_type(eq)           # your existing detector

# (B) Load-or-train the model (5-col CSV: eq1,eq2,surf1,surf2,best_sys)
DATA_FILE  = Path("training_data_MVC_Project.csv")
MODEL_FILE = Path("model.joblib")

def load_or_train():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    if not DATA_FILE.exists():
        # no data yet — start empty
        df = pd.DataFrame(columns=["eq1","eq2","surf1","surf2","best_sys"])
        df.to_csv(DATA_FILE, index=False)
    df = pd.read_csv(DATA_FILE)
    # ensure surf1/surf2 exist
    if "surf1" not in df or "surf2" not in df:
        df["surf1"] = df["eq1"].apply(tag_surface)
        df["surf2"] = df["eq2"].apply(tag_surface)
        df.to_csv(DATA_FILE, index=False)
    X = pd.get_dummies(df[["surf1","surf2"]])
    y = df["best_sys"]
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200,
                                   random_state=42,
                                   class_weight="balanced")
    model.fit(X, y)
    joblib.dump((model, X.columns.tolist()), MODEL_FILE)
    return model, X.columns.tolist()


import sympy as sp
x, y, z, r, theta, rho, phi = sp.symbols('x y z r theta rho phi', real=True)
from sympy import pi
import sympy as sp

# your usual symbols
x,y,z = sp.symbols('x y z', real=True)
r,theta = sp.symbols('r theta', real=True)

def to_cylindrical_limits(f, limits, cylinder_eq=None):
    """
    Convert
        ∫_{x_lo..x_hi} ∫_{y_lo(x)..y_hi(x)} ∫_{z_lo(x,y)..z_hi(x,y)} f(x,y,z) dz dy dx
    into cylindrical
        ∫_{θ=0..2π} ∫_{r=0..r_max(θ)} ∫_{z_lo(r,θ)..z_hi(r,θ)} [ f(r cosθ, r sinθ, z)·r ] dz dr dθ

    Parameters
    ----------
    f            : Sympy expression in (x,y,z)
    limits       : tuple (x_lo, x_hi, y_lo(x), y_hi(x), z_lo(x,y), z_hi(x,y))
    cylinder_eq  : optional Sympy Eq of the form a*x**2 + b*y**2 = C

    Returns
    -------
    f_cyl        : the integrand f(r cosθ, r sinθ, z) * r
    cyl_limits   : (z_lo(r,θ), z_hi(r,θ), r_lo, r_hi(θ), θ_lo, θ_hi)
    """
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = limits

    # 1) Build the cylindrical integrand
    f_sub = f.subs({x: r*sp.cos(theta), y: r*sp.sin(theta)})
    f_cyl = sp.simplify(f_sub * r)

    # 2) Compute z‐bounds in (r,θ)
    z_lo_cyl = sp.simplify(z_lo.subs({x: r*sp.cos(theta), y: r*sp.sin(theta)}))
    z_hi_cyl = sp.simplify(z_hi.subs({x: r*sp.cos(theta), y: r*sp.sin(theta)}))

    # 3) Compute r‐bounds
    if cylinder_eq is not None:
        # exact cylinder: a x^2 + b y^2 = C
        A,B,C0 = [None]*3
        # bring to standard form
        expr = sp.simplify(cylinder_eq.lhs - cylinder_eq.rhs)
        P = sp.Poly(expr, x, y)
        A = P.coeff_monomial(x**2)
        B = P.coeff_monomial(y**2)
        C0 = -P.coeff_monomial(1)
        # r^2 (A cos²θ + B sin²θ) = C0  =>  r ≤ √(C0/(A cos²θ + B sin²θ))
        r_lo = sp.Integer(0)
        r_hi = sp.simplify(sp.sqrt(C0/(A*sp.cos(theta)**2 + B*sp.sin(theta)**2)))
    else:
        # fallback: use the four (x,y) corners of the rectangle
        r_lo = sp.Integer(0)
        corners = [
            (x_lo,      y_lo.subs(x, x_lo)),
            (x_lo,      y_hi.subs(x, x_lo)),
            (x_hi,      y_lo.subs(x, x_hi)),
            (x_hi,      y_hi.subs(x, x_hi))
        ]
        Rcorn = [sp.sqrt(X0**2 + Y0**2) for X0,Y0 in corners]
        r_hi  = sp.simplify(sp.Max(*Rcorn))

    θ_lo, θ_hi = sp.Integer(0), 2*sp.pi

    return f_cyl, (z_lo_cyl, z_hi_cyl, r_lo, r_hi, θ_lo, θ_hi)
def to_spherical_limits(f, limits, sphere_eq=None):
    """
    Convert
      ∫_{x_lo..x_hi} ∫_{y_lo(x)..y_hi(x)} ∫_{z_lo(x,y)..z_hi(x,y)}
         f(x,y,z) dz dy dx
    into spherical
      ∫_{θ=θ_lo..θ_hi} ∫_{φ=φ_lo..φ_hi} ∫_{ρ=ρ_lo..ρ_hi}
         f(ρ sinφ cosθ, ρ sinφ sinθ, ρ cosφ)·ρ^2 sinφ  dρ dφ dθ

    Parameters
    ----------
    f           : Sympy expr in (x,y,z) or a number
    limits      : (x_lo, x_hi, y_lo(x), y_hi(x), z_lo(x,y), z_hi(x,y))
    sphere_eq   : optional Sympy Eq of the form
                  c*(x^2+y^2+z^2) = C  (e.g. x^2+y^2+z^2=R^2)

    Returns
    -------
    f_sph       : integrand f(...) substituted, times rho^2*sin(phi)
    sph_limits  : (rho_lo, rho_hi, phi_lo, phi_hi, theta_lo, theta_hi)
    """
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = limits

    # 1) Make sure f is a Sympy expr
    if isinstance(f, (int,float)):
        f = sp.Integer(f)

    # 2) Substitute (x,y,z) → (ρ sinφ cosθ, ρ sinφ sinθ, ρ cosφ)
    subs_map = {
        x: rho*sp.sin(phi)*sp.cos(theta),
        y: rho*sp.sin(phi)*sp.sin(theta),
        z: rho*sp.cos(phi)
    }
    f_sph = sp.simplify(f.subs(subs_map) * rho**2 * sp.sin(phi))

    # 3) Optional: use exact sphere to get ρ_max
    if sphere_eq is not None:
        expr = sp.simplify(sphere_eq.lhs - sphere_eq.rhs)
        P = sp.Poly(expr, x, y, z)
        # find c*(x^2+y^2+z^2) + c0 = 0
        c2 = P.coeff_monomial(x**2)
        c0 = P.coeff_monomial(1)
        R2 = -c0/c2
        rho_lo = sp.Integer(0)
        rho_hi = sp.simplify(sp.sqrt(R2))

    else:
        # fallback: box corners → max distance from origin
        corners = [
            (x_lo,        y_lo.subs(x, x_lo), z_lo.subs({x:x_lo,y:y_lo.subs(x,x_lo)})),
            (x_lo,        y_lo.subs(x, x_lo), z_hi.subs({x:x_lo,y:y_lo.subs(x,x_lo)})),
            (x_lo,        y_hi.subs(x, x_lo), z_lo.subs({x:x_lo,y:y_hi.subs(x,x_lo)})),
            (x_lo,        y_hi.subs(x, x_lo), z_hi.subs({x:x_lo,y:y_hi.subs(x,x_lo)})),
            (x_hi,        y_lo.subs(x, x_hi), z_lo.subs({x:x_hi,y:y_lo.subs(x,x_hi)})),
            (x_hi,        y_lo.subs(x, x_hi), z_hi.subs({x:x_hi,y:y_lo.subs(x,x_hi)})),
            (x_hi,        y_hi.subs(x, x_hi), z_lo.subs({x:x_hi,y:y_hi.subs(x,x_hi)})),
            (x_hi,        y_hi.subs(x, x_hi), z_hi.subs({x:x_hi,y:y_hi.subs(x,x_hi)}))
        ]
        dists = [sp.sqrt(X**2+Y**2+Z**2) for X,Y,Z in corners]
        rho_lo = sp.Integer(0)
        rho_hi = sp.simplify(sp.Max(*dists))

    # 4) φ and θ always full sweep in absence of more specific surfaces
    phi_lo, phi_hi     = sp.Integer(0), sp.pi
    theta_lo, theta_hi = sp.Integer(0), 2*sp.pi

    return f_sph, (rho_lo, rho_hi, phi_lo, phi_hi, theta_lo, theta_hi)



################################
######        MAIN       #######
################################
def main():
    import sys
    from sympy import simplify, Integral
    
    print("Enter exactly two equations (one per line):")
    eq1_text = input("1) ").strip()
    eq2_text = input("2) ").strip()
    eq1 = parse_equation(eq1_text)
    eq2 = parse_equation(eq2_text)

    t1 = region_type(eq1)
    t2 = region_type(eq2)
    key = frozenset({t1, t2})
    if key not in handlers:
        print(f"Unsupported pair: {t1} & {t2}")
        sys.exit(1)

    func = handlers[key]
    out = func(eq1, eq2)

    function = sp.Integer(1)  # Symbolic integrand for volume
    
    print("\n=== Triple Integral Limits ===")
    if len(out) == 5:
        x_lo, x_hi, y_lo, y_hi, z_segs = out
        print("Rectangular (dz dy dx):")
        print(f"  x from {simplify(x_lo)} to {simplify(x_hi)}")
        print(f" y from {simplify(y_lo)} to {simplify(y_hi)}")
        for i, (z_lo, z_hi) in enumerate(z_segs, start=1):
            print(f"  Segment {i}: z from {simplify(z_lo)} to {simplify(z_hi)}")
            int_rect = Integral(function, (z, z_lo, z_hi), (y, y_lo, y_hi), (x, x_lo, x_hi))
            display(int_rect)
    elif len(out) == 6:
        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = out
        print("Rectangular (dz dy dx):")
        print(f"  x from {simplify(x_lo)} to {simplify(x_hi)}")
        print(f" y from {simplify(y_lo)} to {simplify(y_hi)}")
        print(f" z from {simplify(z_lo)} to {simplify(z_hi)}")
        int_rect = Integral(function, (z, z_lo, z_hi), (y, y_lo, y_hi), (x, x_lo, x_hi))
        display(int_rect)
        
        print("\nCylindrical (dz dr dθ):")
        limits_rect = (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        f_cyl, limits_cyl = to_cylindrical_limits(function, limits_rect)
        z_lo_c, z_hi_c, r_lo, r_hi, theta_lo, theta_hi = limits_cyl
        print(f"  z from {simplify(z_lo_c)} to {simplify(z_hi_c)}")
        print(f" r from {simplify(r_lo)} to {simplify(r_hi)}")
        print(f" θ from {simplify(theta_lo)} to {simplify(theta_hi)}")
        int_cyl = Integral(f_cyl, (z, z_lo_c, z_hi_c), (r, r_lo, r_hi), (theta, theta_lo, theta_hi))
        display(int_cyl)
        
        print("\nSpherical (dρ dφ dθ):")
        f_sph, limits_sph = to_spherical_limits(function, limits_rect)
        rho_lo, rho_hi, phi_lo, phi_hi, theta_lo, theta_hi = limits_sph
        print(f" ρ from {simplify(rho_lo)} to {simplify(rho_hi)}")
        print(f" φ from {simplify(phi_lo)} to {simplify(phi_hi)}")
        print(f" θ from {simplify(theta_lo)} to {simplify(theta_hi)}")
        int_sph = Integral(f_sph, (rho, rho_lo, rho_hi), (phi, phi_lo, phi_hi), (theta, theta_lo, theta_hi))
        display(int_sph)
    else:
        print("Error: handler returned unexpected number of values:", len(out))
        sys.exit(1)

    print(f"\nDetected surface types: {t1} and {t2}")

    if key == frozenset({'sphere', 'cylinder'}):
        if t1 == 'sphere':
            plot_sphere_cylinder(eq1, eq2)
        else:
            plot_sphere_cylinder(eq2, eq1)
    elif key == frozenset({'sphere', 'cone'}):
        if t1 == 'sphere':
            plot_sphere_cone(eq1, eq2)
        else:
            plot_sphere_cone(eq2, eq1)
    elif key == frozenset({'sphere', 'paraboloid'}):
        if t1 == 'sphere':
            plot_sphere_paraboloid(eq1, eq2)
        else:
            plot_sphere_paraboloid(eq2, eq1)
    elif key == frozenset({'sphere', 'sphere'}):
        plot_sphere_sphere(eq1, eq2)
    elif key == frozenset({'sphere', 'plane'}):
        plot_sphere_plane(eq1, eq2)
    elif key == frozenset({'cylinder','cone'}):
        plot_cylinder_cone(eq1, eq2)
    elif key == frozenset({'cylinder','paraboloid'}):
        plot_cylinder_paraboloid(eq1, eq2)
    elif key == frozenset({'cylinder','cylinder'}):
        plot_cylinder_cylinder(eq1, eq2)
    elif key == frozenset({'cone','paraboloid'}):
        plot_cone_paraboloid(eq1, eq2)
    elif key == frozenset({'paraboloid','paraboloid'}):
        plot_paraboloid_paraboloid(eq1, eq2)
    elif key == frozenset({'paraboloid','plane'}):
        plot_paraboloid_plane(eq1, eq2)
    elif key == frozenset({'plane','plane'}):
        plot_plane_plane(eq1, eq2)
    else:
        print(f"Unsupported surface combination: {t1} & {t2}")

    V_exact = compute_volume(eq1, eq2, method='scipy')
    V_mc    = compute_volume(eq1, eq2, method='montecarlo', mc_samples=500_000)
    print("Quadrature volume:", V_exact)
    print("Monte Carlo approx:", V_mc)


    model, DUMMY_COLS = load_or_train()
    print("\n=== Coordinate-System Suggestion ===")
    print(f"Detected surfaces:  {t1} and {t2}")
    df_new = pd.DataFrame([[t1, t2]], columns=["surf1","surf2"])
    X_new  = pd.get_dummies(df_new).reindex(columns=DUMMY_COLS, fill_value=0)
    guess  = model.predict(X_new)[0]
    print(f" I suggest →  {guess}")

    ans = input("Is that correct? [y/n] ").strip().lower()
    if ans in ("n","no"):
        corr = input("What’s the correct system? [rect/cyl/sph] ").strip().lower()
        if corr in {"rect","cyl","sph"}:
            import csv
            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([eq1_text, eq2_text, t1, t2, corr])

            MODEL_FILE.unlink(missing_ok=True)
            model, DUMMY_COLS = load_or_train()
            print(" Thanks, model updated!")
    else:
        print("  Glad it helped!")
    

if __name__ == "__main__":
   
    main()