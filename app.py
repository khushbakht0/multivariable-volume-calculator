import streamlit as st
from sympy import symbols, simplify, Eq, sqrt, solve, Poly
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
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

def parse_equation(s):
    s = s.strip().replace('^','**')
    Ls, Rs = (p.strip() for p in s.split('=',1))
    return Eq(parse_expr(Ls, local_dict=SAFE, transformations=transforms),
              parse_expr(Rs, local_dict=SAFE, transformations=transforms))

def _preprocess_eq(eq):
    pows = list(eq.lhs.atoms(sp.Pow)) + list(eq.rhs.atoms(sp.Pow))
    if any(f.exp == sp.Rational(1,2) for f in pows):
        return sp.Eq(eq.lhs**2, eq.rhs**2)
    return eq

def region_type(eq):
    eq = _preprocess_eq(eq)
    expr = eq.lhs - eq.rhs
    poly = sp.Poly(expr, x, y, z)
    deg = poly.total_degree()
    c_x2 = poly.coeff_monomial(x**2)
    c_y2 = poly.coeff_monomial(y**2)
    c_z2 = poly.coeff_monomial(z**2)
    crosses = sum(poly.coeff_monomial(m) for m in [(1,1,0),(1,0,1),(0,1,1)])
    linears = sum(poly.coeff_monomial(m) for m in [(1,0,0),(0,1,0),(0,0,1)])
    if deg==2 and c_x2==c_y2==c_z2!=0 and crosses==0 and linears==0:
        return 'sphere'
    if deg==2 and crosses==0 and linears==0:
        nz = [v for v in (c_x2, c_y2, c_z2) if v!=0]
        zc = [v for v in (c_x2, c_y2, c_z2) if v==0]
        if len(nz)==2 and len(zc)==1:
            return 'cylinder'
    if deg==2 and poly.degree(z)==1 and any(poly.coeff_monomial(m)!=0 for m in [(2,0,0),(0,2,0)]):
        return 'paraboloid'
    if deg==2 and poly.degree(z)==2 and crosses==0:
        if poly.coeff_monomial((1,0,0))==0 and poly.coeff_monomial((0,1,0))==0:
            return 'cone'
    if deg==1:
        return 'plane'
    raise ValueError("Cannot identify region type")

def safe_lambdify(sym_expr, args):
    def clamp_sqrt(u):
        return np.sqrt(np.clip(np.array(u), 0, None))
    modules = [{'sqrt': clamp_sqrt, 'abs': np.abs}, 'numpy']
    f = sp.lambdify(args, sym_expr, modules=modules)
    def wrapper(*nums):
        with np.errstate(invalid='ignore', divide='ignore'):
            out = f(*nums)
        out = np.real(out)
        return np.where(np.isfinite(out), out, np.nan)
    return wrapper

def handle_sphere_cylinder(eqA, eqB):
    def is_sphere(eq):
        p = Poly(eq.lhs - eq.rhs, x, y, z)
        return (p.total_degree()==2 and p.coeff_monomial(x**2)==p.coeff_monomial(y**2)==p.coeff_monomial(z**2)
                and all(p.coeff_monomial(m)==0 for m in [(1,1,0),(1,0,1),(0,1,1),(1,0,0),(0,1,0),(0,0,1)]))
    if is_sphere(eqA): sph_eq, cyl_eq = eqA, eqB
    else: sph_eq, cyl_eq = eqB, eqA
    sph_eq = _preprocess_eq(sph_eq)
    Ps = Poly(sph_eq.lhs - sph_eq.rhs, x, y, z)
    R2 = simplify(-Ps.coeff_monomial(1) / Ps.coeff_monomial(x**2))
    cyl_eq = _preprocess_eq(cyl_eq)
    Pc = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = Pc.coeff_monomial(x**2)
    b = Pc.coeff_monomial(y**2)
    C = simplify(-Pc.coeff_monomial(1))
    return (-sqrt(C/a), sqrt(C/a), -sqrt((C-a*x**2)/b), sqrt((C-a*x**2)/b), -sqrt(R2-x**2-y**2), sqrt(R2-x**2-y**2))

def handle_sphere_cone(sph_eq, cone_eq):
    P = sp.Poly(sph_eq.lhs - sph_eq.rhs, x, y, z)
    R2 = -P.coeff_monomial(1)/P.coeff_monomial(x**2)
    r2 = R2/2
    return (-sp.sqrt(r2), sp.sqrt(r2), -sp.sqrt(r2-x**2), sp.sqrt(r2-x**2), sp.sqrt(x**2+y**2), sp.sqrt(R2-x**2-y**2))

def handle_sphere_plane(eq1, eq2):
    def _is_sphere(e):
        P = Poly(e.lhs - e.rhs, x,y,z)
        c2 = P.coeff_monomial(x**2)
        return (P.total_degree()==2 and c2!=0 and P.coeff_monomial(y**2)==c2 and P.coeff_monomial(z**2)==c2
                and all(P.coeff_monomial(m)==0 for m in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]))
    if _is_sphere(eq1): sph_eq, plane_eq = eq1, eq2
    else: sph_eq, plane_eq = eq2, eq1
    R2 = simplify(solve(sph_eq, z**2)[0].subs({x:0,y:0}))
    expr = plane_eq.lhs - plane_eq.rhs
    Pp = Poly(expr, x,y,z)
    A = Pp.coeff_monomial(x)
    B = Pp.coeff_monomial(y)
    C = Pp.coeff_monomial(z)
    D = -Pp.coeff_monomial(1)
    if C == 0:
        return (-sqrt(R2), sqrt(R2), -sqrt(R2-x**2), sqrt(R2-x**2), -sqrt(R2-x**2-y**2), sqrt(R2-x**2-y**2))
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
    return (x_lo, x_hi, y_lo, y_hi, simplify(sp.Min(z_plane, z_sph)), simplify(sp.Max(z_plane, z_sph)))

def handle_sphere_paraboloid(sph_eq, para_eq):
    R2 = simplify(solve(sph_eq, z**2)[0].subs({x:0,y:0}))
    z_para = simplify(solve(para_eq, z)[0])
    Pxy = Poly(z_para, x, y)
    a = simplify(Pxy.coeff_monomial(x**2))
    c = simplify(Pxy.coeff_monomial(1))
    roots = solve((a*u + c)**2 + u - R2, u)
    up = [s for s in roots if s.is_real and s>0]
    u_pos = simplify(up[0])
    return (-sqrt(u_pos), sqrt(u_pos), -sqrt(u_pos-x**2), sqrt(u_pos-x**2), z_para, sqrt(R2-x**2-y**2))

def handle_cylinder_cone(eqA, eqB):
    if region_type(eqA) == 'cylinder': cyl_eq, cone_eq = eqA, eqB
    else: cyl_eq, cone_eq = eqB, eqA
    Pc = sp.Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = Pc.coeff_monomial(x**2)
    b = Pc.coeff_monomial(y**2)
    C = simplify(-Pc.coeff_monomial(1))
    Q = simplify(solve(cone_eq, z**2)[0])
    return (-sqrt(C/a), sqrt(C/a), -sqrt((C-a*x**2)/b), sqrt((C-a*x**2)/b), -sqrt(Q), sqrt(Q))

def handle_cylinder_plane(eqA, eqB):
    if region_type(eqA) == 'cylinder': cyl_eq, plane_eq = eqA, eqB
    else: cyl_eq, plane_eq = eqB, eqA
    Pc = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = Pc.coeff_monomial(x**2)
    b = Pc.coeff_monomial(y**2)
    C = simplify(-Pc.coeff_monomial(1))
    z_plane = simplify(solve(plane_eq, z)[0])
    z_lo, z_hi = 0, z_plane
    if float(z_hi.subs({x:0,y:0})) < 0: z_lo, z_hi = z_hi, z_lo
    return (-sqrt(C/a), sqrt(C/a), -sqrt((C-a*x**2)/b), sqrt((C-a*x**2)/b), z_lo, z_hi)

def handle_cylinder_paraboloid(eqA, eqB):
    if region_type(eqA) == 'cylinder': cyl_eq, para_eq = eqA, eqB
    else: cyl_eq, para_eq = eqB, eqA
    P = Poly(cyl_eq.lhs - cyl_eq.rhs, x, y)
    a = P.coeff_monomial(x**2)
    b = P.coeff_monomial(y**2)
    C = simplify(-P.coeff_monomial(1))
    f = simplify(solve(para_eq, z)[0])
    base, top = 0, f
    if top.subs({x:0,y:0}) < base: base, top = top, base
    return (-sqrt(C/a), sqrt(C/a), -sqrt((C-a*x**2)/b), sqrt((C-a*x**2)/b), base, top)

def handle_cone_paraboloid(eqA, eqB):
    if region_type(eqA) == 'cone': cone_eq, para_eq = eqA, eqB
    else: cone_eq, para_eq = eqB, eqA
    z_para = simplify(solve(para_eq, z)[0])
    Pp = Poly(z_para, x, y)
    a = simplify(Pp.coeff_monomial(x**2))
    c = simplify(Pp.coeff_monomial(1))
    Q = simplify(solve(cone_eq, z**2)[0])
    Pc = Poly(Q, x, y)
    A = simplify(Pc.coeff_monomial(x**2))
    roots = solve(a**2*u**2 + (2*a*c - A)*u + c**2, u)
    us = [r for r in roots if r.is_real and r > 0]
    u0 = simplify(us[0])
    if c <= 0: z_lo, z_hi = z_para, sqrt(A*(x**2+y**2))
    else: z_lo, z_hi = -sqrt(A*(x**2+y**2)), z_para
    return (-sqrt(u0), sqrt(u0), -sqrt(u0-x**2), sqrt(u0-x**2), z_lo, z_hi)

def handle_paraboloid_paraboloid(eq1, eq2):
    z1 = simplify(solve(eq1, z)[0])
    z2 = simplify(solve(eq2, z)[0])
    D = simplify(z1 - z2)
    P = Poly(D, x, y)
    a = P.coeff_monomial(x**2)
    c0 = P.coeff_monomial(1)
    R2 = simplify(-c0/a)
    R = sqrt(R2)
    if float(z1.subs({x:0,y:0})) < float(z2.subs({x:0,y:0})): z_lo, z_hi = z1, z2
    else: z_lo, z_hi = z2, z1
    return (-R, R, -sqrt(R2-x**2), sqrt(R2-x**2), z_lo, z_hi)

def handle_plane_plane(eq1, eq2):
    z1 = solve(eq1, z)[0]
    z2 = solve(eq2, z)[0]
    if z1.subs({x:0,y:0}) < z2.subs({x:0,y:0}): z_lo, z_hi = z1, z2
    else: z_lo, z_hi = z2, z1
    expr = simplify((eq1.lhs - eq1.rhs).subs(z,0))
    A = expr.coeff(x)
    B = expr.coeff(y)
    Cv = -expr.subs({x:0,y:0})
    return (0, simplify(Cv/A), 0, simplify((Cv-A*x)/B), z_lo, z_hi)

handlers = {
    frozenset({'sphere','cylinder'}): handle_sphere_cylinder,
    frozenset({'sphere','cone'}): handle_sphere_cone,
    frozenset({'sphere','plane'}): handle_sphere_plane,
    frozenset({'sphere','paraboloid'}): handle_sphere_paraboloid,
    frozenset({'cylinder','cone'}): handle_cylinder_cone,
    frozenset({'cylinder','plane'}): handle_cylinder_plane,
    frozenset({'cylinder','paraboloid'}): handle_cylinder_paraboloid,
    frozenset({'cone','paraboloid'}): handle_cone_paraboloid,
    frozenset({'paraboloid','paraboloid'}): handle_paraboloid_paraboloid,
    frozenset({'plane','plane'}): handle_plane_plane,
}

def compute_volume(eq1, eq2):
    types = frozenset({region_type(eq1), region_type(eq2)})
    if types not in handlers:
        return None, f"No handler for {types}"
    x_lo_s, x_hi_s, y_lo_s, y_hi_s, z_lo_s, z_hi_s = handlers[types](eq1, eq2)
    x_lo, x_hi = float(x_lo_s), float(x_hi_s)
    fy_lo = safe_lambdify(y_lo_s, (x,))
    fy_hi = safe_lambdify(y_hi_s, (x,))
    fz_lo = safe_lambdify(z_lo_s, (x, y))
    fz_hi = safe_lambdify(z_hi_s, (x, y))
    def integrand(yv, xv):
        dz = fz_hi(xv, yv) - fz_lo(xv, yv)
        return np.where((dz>0)&np.isfinite(dz), dz, 0.0)
    V, _ = integrate.dblquad(integrand, x_lo, x_hi, lambda xv: fy_lo(xv), lambda xv: fy_hi(xv), epsabs=1e-6, epsrel=1e-6)
    return V, None

st.sidebar.title("Examples")
st.sidebar.code("Sphere x Cylinder:\nx**2+y**2+z**2=25\nx**2+y**2=9")
st.sidebar.code("Sphere x Cone:\nx**2+y**2+z**2=16\nz**2=x**2+y**2")
st.sidebar.code("Cylinder x Paraboloid:\nx**2+y**2=4\nz=x**2+y**2")
st.sidebar.code("Two Paraboloids:\nz=x**2+y**2\nz=8-x**2-y**2")
st.sidebar.code("Sphere x Plane:\nx**2+y**2+z**2=9\nx+y+z=3")
st.sidebar.code("Two Planes:\nx+y+z=2\n2x+2y+z=4")

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
