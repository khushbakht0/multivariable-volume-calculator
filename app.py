import streamlit as st
import sys
from sympy import symbols, simplify, Eq, Interval, sqrt, solve
from sympy.polys.polytools import Poly
from sympy.solvers.inequalities import solve_univariate_inequality
import sympy as sp
from sympy import (
    symbols, Eq, solve, simplify,
    solveset, S,
    sqrt, Poly, lambdify
)
import math
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application
)
import numpy as np
from scipy import integrate

st.set_page_config(page_title="MVC Volume Calculator", layout="wide")
st.title("Multivariable Calculus - Volume Calculator")
st.markdown("Enter two surface equations to compute the bounded volume between them.")

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
    pows = list(eq.lhs.atoms(sp.Pow)) + list(eq.rhs.atoms(sp.Pow))
    if any(f.exp == sp.Rational(1,2) for f in pows):
        return sp.Eq(eq.lhs**2, eq.rhs**2)
    return eq

def region_type(eq: sp.Eq) -> str:
    eq = _preprocess_eq(eq)
    expr = eq.lhs - eq.rhs
    poly = sp.Poly(expr, x, y, z)
    deg = poly.total_degree()
    c_x2 = poly.coeff_monomial(x**2)
    c_y2 = poly.coeff_monomial(y**2)
    c_z2 = poly.coeff_monomial(z**2)
    crosses = sum(poly.coeff_monomial(m) for m in [(1,1,0),(1,0,1),(0,1,1)])
    linears = sum(poly.coeff_monomial(m) for m in [(1,0,0),(0,1,0),(0,0,1)])
    if (deg==2 and c_x2==c_y2==c_z2!=0 and crosses==0 and linears==0):
        return 'sphere'
    if (deg==2 and crosses==0 and linears==0):
        nz = [v for v in (c_x2, c_y2, c_z2) if v!=0]
        zc = [v for v in (c_x2, c_y2, c_z2) if v==0]
        if len(nz)==2 and len(zc)==1:
            return 'cylinder'
    if deg==2 and poly.degree(z)==1 and any(poly.coeff_monomial(m)!=0 for m in [(2,0,0),(0,2,0)]):
        return 'paraboloid'
    if deg==2 and poly.degree(z)==2:
        if crosses==0 and poly.coeff_monomial((1,0,0))==0 and poly.coeff_monomial((0,1,0))==0:
            return 'cone'
    if deg==1:
        return 'plane'
    raise ValueError(f"Cannot identify region type for {eq}")

def safe_lambdify(sym_expr, args):
    def clamp_sqrt(u):
        u = np.array(u)
        return np.sqrt(np.clip(u, 0.0, None))
    modules = [{'sqrt': clamp_sqrt, 'abs': np.abs, 'log': np.log,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'exp': np.exp}, 'numpy']
    f = sp.lambdify(args, sym_expr, modules=modules)
    def f_safe(*nums):
        with np.errstate(invalid='ignore', divide='ignore'):
            out = f(*nums)
        out = np.real(out)
        return np.where(np.isfinite(out), out, np.nan)
    return f_safe

def handle_sphere_cylinder(eqA, eqB):
    def is_sphere(eq):
        p = Poly(eq.lhs - eq.rhs, x, y, z)
        return (p.total_degree()==2 and p.coeff_monomial(x**2)==p.coeff_monomial(y**2)==p.coeff_monomial(z**2)
                and all(p.coeff_monomial(mon)==0 for mon in [(1,1,0),(1,0,1),(0,1,1),(1,0,0),(0,1,0),(0,0,1)]))
    if is_sphere(eqA): sph_eq, cyl_eq = eqA, eqB
    else: sph_eq, cyl_eq = eqB, eqA
    sph_eq = _preprocess_eq(sph_eq)
    Ps = Poly(sph_eq.lhs - sph_eq.rhs, x, y, z)
    c_s = Ps.coeff_monomial(x**2)
    c0_s = Ps.coeff_monomial(1)
    R2 = simplify(-c0_s / c_s)
    cyl_eq = _preprocess_eq(cyl_eq)
    Pc = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = Pc.coeff_monomial(x**2)
    b = Pc.coeff_monomial(y**2)
    c0_c = Pc.coeff_monomial(1)
    C = simplify(-c0_c)
    x_lo = -sqrt(C/a)
    x_hi = sqrt(C/a)
    y_lo = -sqrt((C - a*x**2)/b)
    y_hi = sqrt((C - a*x**2)/b)
    z_lo = -sqrt(R2 - x**2 - y**2)
    z_hi = sqrt(R2 - x**2 - y**2)
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo), simplify(z_hi)

def handle_sphere_cone(sph_eq, cone_eq):
    P = sp.Poly(sph_eq.lhs - sph_eq.rhs, x, y, z)
    c0 = P.coeff_monomial(1)
    c_x2 = P.coeff_monomial(x**2)
    R2 = -c0/c_x2
    r2 = R2/2
    x_lo, x_hi = -sp.sqrt(r2), sp.sqrt(r2)
    y_lo = -sp.sqrt(r2 - x**2)
    y_hi = sp.sqrt(r2 - x**2)
    z_lo = sp.sqrt(x**2 + y**2)
    z_hi = sp.sqrt(R2 - x**2 - y**2)
    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi

def handle_cylinder_cone(eqA, eqB):
    if region_type(eqA) == 'cylinder': cyl_eq, cone_eq = eqA, eqB
    else: cyl_eq, cone_eq = eqB, eqA
    Pc = sp.Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = Pc.coeff_monomial(x**2)
    b = Pc.coeff_monomial(y**2)
    C = simplify(-Pc.coeff_monomial(1))
    sol_z2 = solve(cone_eq, z**2)
    Q = simplify(sol_z2[0])
    x_lo, x_hi = -sqrt(C/a), sqrt(C/a)
    y_lo, y_hi = -sqrt((C - a*x**2)/b), sqrt((C - a*x**2)/b)
    z_lo, z_hi = -sqrt(Q), sqrt(Q)
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo), simplify(z_hi)

def handle_cylinder_cylinder(eq1, eq2):
    P1 = Poly(eq1.lhs - eq1.rhs, x, y, z)
    P2 = Poly(eq2.lhs - eq2.rhs, x, y, z)
    if P1.coeff_monomial(z**2) == 0 and (P1.coeff_monomial(x**2) or P1.coeff_monomial(y**2)):
        Pv, Pp = P1, P2
    else:
        Pv, Pp = P2, P1
    a = Pv.coeff_monomial(x**2)
    b = Pv.coeff_monomial(y**2)
    C = simplify(-Pv.coeff_monomial(1))
    p_y2 = Pp.coeff_monomial(y**2)
    p_z2 = Pp.coeff_monomial(z**2)
    p_x2 = Pp.coeff_monomial(x**2)
    C2 = simplify(-Pp.coeff_monomial(1))
    if p_y2 and p_z2:
        z_lo = -sqrt((C2 - p_y2*y**2)/p_z2)
        z_hi = sqrt((C2 - p_y2*y**2)/p_z2)
    else:
        z_lo = -sqrt((C2 - p_x2*x**2)/p_z2)
        z_hi = sqrt((C2 - p_x2*x**2)/p_z2)
    x_lo = -sqrt(C/a)
    x_hi = sqrt(C/a)
    y_lo = -sqrt((C - a*x**2)/b)
    y_hi = sqrt((C - a*x**2)/b)
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo), simplify(z_hi)

def handle_cone_paraboloid(eqA, eqB):
    if region_type(eqA) == 'cone': cone_eq, para_eq = eqA, eqB
    else: cone_eq, para_eq = eqB, eqA
    psol = solve(para_eq, z)
    z_para = simplify(psol[0])
    Pp = Poly(z_para, x, y)
    a = simplify(Pp.coeff_monomial(x**2))
    c = simplify(Pp.coeff_monomial(1))
    csol = solve(cone_eq, z**2)
    Q = simplify(csol[0])
    Pc = Poly(Q, x, y)
    A = simplify(Pc.coeff_monomial(x**2))
    quad = simplify(a**2*u**2 + (2*a*c - A)*u + c**2)
    roots = solve(quad, u)
    us = [r for r in roots if r.is_real and r > 0]
    u0 = simplify(us[0])
    x_lo, x_hi = -sqrt(u0), sqrt(u0)
    y_lo, y_hi = -sqrt(u0 - x**2), sqrt(u0 - x**2)
    if c <= 0: z_lo, z_hi = z_para, sqrt(A*(x**2 + y**2))
    else: z_lo, z_hi = -sqrt(A*(x**2 + y**2)), z_para
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo), simplify(z_hi)

def handle_sphere_plane(eq1, eq2):
    def _is_sphere(e):
        P = Poly(e.lhs - e.rhs, x,y,z)
        c2 = P.coeff_monomial(x**2)
        return (P.total_degree()==2 and c2!=0 and P.coeff_monomial(y**2)==c2 and P.coeff_monomial(z**2)==c2
                and all(P.coeff_monomial(m)==0 for m in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]))
    if _is_sphere(eq1): sph_eq, plane_eq = eq1, eq2
    else: sph_eq, plane_eq = eq2, eq1
    sol_z2 = solve(sph_eq, z**2)
    R2 = simplify(sol_z2[0].subs({x:0,y:0}))
    expr = plane_eq.lhs - plane_eq.rhs
    Pp = Poly(expr, x,y,z)
    A = Pp.coeff_monomial(x)
    B = Pp.coeff_monomial(y)
    C = Pp.coeff_monomial(z)
    D = -Pp.coeff_monomial(1)
    if C == 0:
        if B==0: x_lo, x_hi = simplify(D/A), sqrt(R2)
        else: x_lo, x_hi = 0, sqrt(R2)
        y_lo = -sqrt(R2 - x**2)
        y_hi = sqrt(R2 - x**2)
        z_lo = -sqrt(R2 - x**2 - y**2)
        z_hi = sqrt(R2 - x**2 - y**2)
        return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
    z_plane = simplify((D - A*x - B*y)/C)
    E = simplify(x**2 + y**2 + z_plane**2 - R2)
    xs0 = solve(E.subs(y,0), x)
    reals_x = sorted([s for s in xs0 if s.is_real], key=lambda s: float(s))
    x_lo, x_hi = simplify(reals_x[0]), simplify(reals_x[1])
    Ey = Poly(E, y)
    Ay = Ey.coeff_monomial(y**2)
    By = Ey.coeff_monomial(y)
    Cy = Ey.coeff_monomial(1)
    disc = simplify(By**2 - 4*Ay*Cy)
    sd = sqrt(disc)
    y_lo = simplify((-By - sd)/(2*Ay))
    y_hi = simplify((-By + sd)/(2*Ay))
    z_sph = sqrt(R2 - x**2 - y**2)
    z_lo = simplify(sp.Min(z_plane, z_sph))
    z_hi = simplify(sp.Max(z_plane, z_sph))
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo), simplify(z_hi)

def handle_sphere_paraboloid(sph_eq, para_eq):
    sol_z2 = solve(sph_eq, z**2)
    R2 = simplify(sol_z2[0].subs({x:0, y:0}))
    para_sols = solve(para_eq, z)
    z_para = simplify(para_sols[0])
    Pxy = Poly(z_para, x, y)
    a = simplify(Pxy.coeff_monomial(x**2))
    c = simplify(Pxy.coeff_monomial(1))
    u = symbols('u', real=True)
    quad = Poly((a*u + c)**2 + u - R2, u)
    roots = solve(quad.as_expr(), u)
    up = [s for s in roots if s.is_real and s>0]
    u_pos = simplify(up[0])
    x_lo, x_hi = -sqrt(u_pos), sqrt(u_pos)
    y_lo = -sqrt(u_pos - x**2)
    y_hi = sqrt(u_pos - x**2)
    z_lo = z_para
    z_hi = sqrt(R2 - x**2 - y**2)
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo), simplify(z_hi)

def handle_cylinder_plane(eqA, eqB):
    if region_type(eqA) == 'cylinder': cyl_eq, plane_eq = eqA, eqB
    else: cyl_eq, plane_eq = eqB, eqA
    Pc = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = Pc.coeff_monomial(x**2)
    b = Pc.coeff_monomial(y**2)
    C = simplify(-Pc.coeff_monomial(1))
    x_lo = -sqrt(C/a)
    x_hi = sqrt(C/a)
    y_lo = -sqrt((C - a*x**2)/b)
    y_hi = sqrt((C - a*x**2)/b)
    z_sols = solve(plane_eq, z)
    z_plane = simplify(z_sols[0])
    z_lo, z_hi = 0, z_plane
    if float(z_hi.subs({x:0,y:0})) < float(z_lo):
        z_lo, z_hi = z_hi, z_lo
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo), simplify(z_hi)

def handle_cylinder_paraboloid(eqA, eqB):
    if region_type(eqA) == 'cylinder': cyl_eq, para_eq = eqA, eqB
    else: cyl_eq, para_eq = eqB, eqA
    P = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = P.coeff_monomial(x**2)
    b = P.coeff_monomial(y**2)
    C = simplify(-P.coeff_monomial(1))
    sols = solve(para_eq, z)
    f = simplify(sols[0])
    x_lo = -sqrt(C/a)
    x_hi = sqrt(C/a)
    y_lo = -sqrt((C - a*x**2)/b)
    y_hi = sqrt((C - a*x**2)/b)
    base, top = 0, f
    if (top.subs({x:0,y:0}) < base): base, top = top, base
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(base), simplify(top)

def handle_paraboloid_paraboloid(eq1, eq2):
    sols1 = solve(eq1, z)
    sols2 = solve(eq2, z)
    z1 = simplify(sols1[0])
    z2 = simplify(sols2[0])
    Δ = simplify(z1 - z2)
    P = Poly(Δ, x, y)
    a = P.coeff_monomial(x**2)
    c0 = P.coeff_monomial(1)
    R2 = simplify(-c0 / a)
    R = sqrt(R2)
    if float(z1.subs({x:0,y:0})) < float(z2.subs({x:0,y:0})):
        z_lo_sym, z_hi_sym = z1, z2
    else:
        z_lo_sym, z_hi_sym = z2, z1
    x_lo, x_hi = -R, R
    y_lo = -sqrt(R2 - x**2)
    y_hi = sqrt(R2 - x**2)
    return simplify(x_lo), simplify(x_hi), simplify(y_lo), simplify(y_hi), simplify(z_lo_sym), simplify(z_hi_sym)

def handle_plane_plane(eq1, eq2):
    z1 = solve(eq1, z)[0]
    z2 = solve(eq2, z)[0]
    if z1.subs({x:0, y:0}) < z2.subs({x:0, y:0}): z_lo, z_hi = z1, z2
    else: z_lo, z_hi = z2, z1
    expr1 = simplify((eq1.lhs - eq1.rhs).subs(z, 0))
    A = expr1.coeff(x)
    B = expr1.coeff(y)
    K = expr1.subs({x:0, y:0})
    C_val = -K
    x_lo, x_hi = 0, simplify(C_val/A)
    y_lo = 0
    y_hi = simplify((C_val - A*x)/B)
    return x_lo, x_hi, y_lo, y_hi, simplify(z_lo), simplify(z_hi)

def find_z_bounds(surfs):
    if len(surfs) == 1:
        sols = solve(surfs[0], z)
        sols = sorted(sols, key=lambda f: float(f.subs({x:0,y:0}).evalf()))
        return simplify(sols[0]), simplify(sols[1])
    elif len(surfs) == 2:
        s0 = solve(surfs[0], z)[0]
        s1 = solve(surfs[1], z)[0]
        sols = sorted([s0, s1], key=lambda f: float(f.subs({x:0,y:0}).evalf()))
        return simplify(sols[0]), simplify(sols[1])

def find_xy_bounds(surfs, cons, z_bounds):
    x, y, z = symbols('x y z', real=True)
    z_lo, z_hi = z_bounds
    if len(cons)==1 and len(surfs)==1:
        c = cons[0]
        y_sols = solve(c, y)
        y_vals = [float(sol.subs(x,0)) for sol in y_sols]
        y_lo = y_sols[y_vals.index(min(y_vals))]
        y_hi = y_sols[y_vals.index(max(y_vals))]
        x_sols = solve(c.subs(y,0), x)
        x_vals = [float(sol) for sol in x_sols]
        x_lo = x_sols[x_vals.index(min(x_vals))]
        x_hi = x_sols[x_vals.index(max(x_vals))]
        return (y_lo, y_hi), (x_lo, x_hi)
    if len(cons)==0 and len(surfs)==2:
        Δ = simplify(z_hi - z_lo)
        y_branches = solve(Δ, y)
        y1, y2 = [simplify(sol) for sol in y_branches]
        P = Poly(Δ, y, x)
        A = P.coeff_monomial(y**2)
        B = P.coeff_monomial(y**1)
        C = P.coeff_monomial(1)
        D = simplify(B**2 - 4*A*C)
        x_region = solve_univariate_inequality(D >= 0, x)
        if isinstance(x_region, Interval):
            x_lo, x_hi = x_region.start, x_region.end
        else:
            ivs = [iv for iv in x_region.args if isinstance(iv, Interval)]
            x_lo, x_hi = min(iv.start for iv in ivs), max(iv.end for iv in ivs)
        x_mid = (x_lo + x_hi)/2
        if float(y1.subs(x, x_mid)) < float(y2.subs(x, x_mid)):
            y_lo, y_hi = y1, y2
        else:
            y_lo, y_hi = y2, y1
        return (simplify(y_lo), simplify(y_hi)), (simplify(x_lo), simplify(x_hi))

handlers = {
    frozenset({'cylinder','paraboloid'}): handle_cylinder_paraboloid,
    frozenset({'paraboloid','paraboloid'}): handle_paraboloid_paraboloid,
    frozenset({'plane','plane'}): handle_plane_plane,
    frozenset({'sphere','cone'}): handle_sphere_cone,
    frozenset({'sphere','paraboloid'}): handle_sphere_paraboloid,
    frozenset({'sphere','cylinder'}): handle_sphere_cylinder,
    frozenset({'sphere','plane'}): handle_sphere_plane,
    frozenset({'cylinder','cone'}): handle_cylinder_cone,
    frozenset({'cylinder','cylinder'}): handle_cylinder_cylinder,
    frozenset({'cone','paraboloid'}): handle_cone_paraboloid,
    frozenset({'cylinder','plane'}): handle_cylinder_plane,
}

def compute_volume(eq1, eq2):
    types = frozenset({region_type(eq1), region_type(eq2)})
    if types not in handlers:
        return None, f"No handler for shapes {types}"
    handler = handlers[types]
    out = handler(eq1, eq2)
    if len(out) == 5:
        x_lo_s, x_hi_s, y_lo_s, y_hi_s, z_segs = out
        z_lo_s, z_hi_s = z_segs[0]
    else:
        x_lo_s, x_hi_s, y_lo_s, y_hi_s, z_lo_s, z_hi_s = out
    x_lo, x_hi = float(x_lo_s), float(x_hi_s)
    f_y_lo = safe_lambdify(y_lo_s, (x,))
    f_y_hi = safe_lambdify(y_hi_s, (x,))
    f_z_lo = safe_lambdify(z_lo_s, (x, y))
    f_z_hi = safe_lambdify(z_hi_s, (x, y))
    def integrand(y_val, x_val):
        dz = f_z_hi(x_val, y_val) - f_z_lo(x_val, y_val)
        return np.where((dz>0)&np.isfinite(dz), dz, 0.0)
    V, _ = integrate.dblquad(integrand, x_lo, x_hi,
                              lambda xv: f_y_lo(xv), lambda xv: f_y_hi(xv),
                              epsabs=1e-6, epsrel=1e-6)
    return V, None

col1, col2 = st.columns(2)
with col1:
    eq1_text = st.text_input("Equation 1", value="x**2 + y**2 + z**2 = 25")
with col2:
    eq2_text = st.text_input("Equation 2", value="x**2 + y**2 = 9")

if st.button("Compute Volume", type="primary"):
    try:
        eq1 = parse_equation(eq1_text)
        eq2 = parse_equation(eq2_text)
        t1, t2 = region_type(eq1), region_type(eq2)
        V, error = compute_volume(eq1, eq2)
        if error:
            st.error(error)
        else:
            st.success(f"Volume = {V:.6f}")
            st.info(f"Surfaces detected: {t1} and {t2}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
