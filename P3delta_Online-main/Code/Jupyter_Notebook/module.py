import sys
from IPython.core.display import display, HTML
import numpy as np
import sympy as sp
import cxroots as cx
import scipy.special as spsp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
from ipywidgets import AppLayout
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML, Javascript, FileLink
import traceback
from fpdf import FPDF
from urllib.request import urlopen
import os
from os import path
from PIL import Image
from matplotlib.offsetbox import (OffsetImage,AnchoredOffsetbox)

# -----------------------------------------------------------------------------
# FUNCTION FOR BACK-END
# -----------------------------------------------------------------------------
# Plot

def watermark2(ax, fig):
    img = Image.open("images/logoP3D.png")
    width, height = ax.figure.get_size_inches() * fig.dpi
    wm_width = int(width / 4)
    scaling = (wm_width / float(img.size[0]))
    wm_height = int(float(img.size[1]) * float(scaling))
    img = img.resize((wm_width, wm_height), Image.ANTIALIAS)
    imagebox = OffsetImage(img, alpha=0.2)
    imagebox.image.axes = ax
    ao = AnchoredOffsetbox(loc='center', pad=0.01, borderpad=0, child=imagebox)
    ao.patch.set_alpha(0)
    ax.add_artist(ao)
    
# -----------------------------------------------------------------------------
# Generic MID tab


def compute_root_spectrum_mid_generic(n, m, s0, value_tau):
    s = sp.symbols('s')
    tau = sp.symbols('tau')
    a = sp.symbols(["a{:d}".format(i) for i in range(n)], real=True)
    alpha = sp.symbols(["alpha{:d}".format(i) for i in range(m + 1)],
                       real=True)
    polynomial = s ** n + np.array(a).dot([s ** i for i in range(n)])
    delayed = np.array(alpha).dot([s ** i for i in range(m + 1)]) * sp.exp(
        -s * tau)
    q = polynomial + delayed
    sysderivatif = [q]
    for i in range(n + m + 1):
        dernierederivee = sysderivatif[-1]
        sysderivatif.append(dernierederivee.diff(s))
    sol = sp.linsolve(sysderivatif[:-1], alpha + a).args[0]
    solnum = sol.subs({s: s0})
    solnum = solnum.subs({tau: value_tau})
    a_num = list(solnum[m + 1:])
    a_coeff_mid_generic = a_num.copy()
    a_coeff_mid_generic.append(1)
    alpha_num = list(solnum[:m + 1])
    b_coeff_mid_generic = alpha_num.copy()
    qnumerique = s ** n + np.array(a_num).dot([s ** i for i in range(n)]) + \
        np.array(alpha_num).dot([s ** i for i in range(m + 1)]) * \
        sp.exp(-s * tau)
    qnumerique = qnumerique.subs(tau, value_tau)
    sysrootfinding = [qnumerique, qnumerique.diff(s)]
    sysfunc = [sp.lambdify(s, i) for i in sysrootfinding]
    rect = cx.Rectangle([-100, 10], [-100, 100])
    roots = rect.roots(sysfunc[0], sysfunc[1], rootErrTol=1e-5, absTol=1e-5,
                       M=n + m + 1)
    xroot = np.real(roots[0])
    yroot = np.imag(roots[0])
    return xroot, yroot, qnumerique, a_coeff_mid_generic, b_coeff_mid_generic


def factorization_integral_latex(n, m, s0, tau):
    factor = str(tau ** (m + 1) / spsp.factorial(m))
    parenthesis = "(s + " + str(-s0) + ")"
    power = "^{" + str(n + m + 1) + "}"
    return r"\$" + "\\Delta(s) = " + factor + parenthesis + power + \
           r"\int_0^1 t^{" + str(m) + r"} (1 - t)^{" + str(n) + \
           "} e^{-" + str(tau) + "t" + parenthesis + \
           "} \\mathrm{d}t" + "$"


def factorization_1f1_latex(n, m, s0, tau):
    factor = str(
        tau ** (m + 1) * spsp.factorial(n) / spsp.factorial(n + m + 1))
    parenthesis = "(s + " + str(-s0) + ")"
    power = "^{" + str(n + m + 1) + "}"
    return r"\$" + "\\Delta(s) = " + factor + parenthesis + power + \
           r" {}_1 F_1(" + str(m + 1) + r", " + str(n + m + 2) + ", -" + \
           str(tau) + parenthesis + ")" + "$"


# -----------------------------------------------------------------------------
# Control oriented MID tab


def compute_admissibilite(n, m, a_coeff_mid_co):
    s = sp.symbols('s')
    tau = sp.symbols('tau')
    a = sp.symbols(["a{:d}".format(i) for i in range(n)], real=True)
    alpha = sp.symbols(["alpha{:d}".format(i) for i in range(m + 1)],
                       real=True)
    avalue = a_coeff_mid_co
    polynomial = s ** n + np.array(a).dot([s ** i for i in range(n)])
    delayed = np.array(alpha).dot([s ** i for i in range(m + 1)]) * sp.exp(
        -s * tau)
    q = polynomial + delayed
    sysderivatif = [q]
    for i in range(m + 1):
        dernierederivee = sysderivatif[-1]
        sysderivatif.append(dernierederivee.diff(s))
    sol = sp.linsolve(sysderivatif[:-1], alpha).args[0]
    polyadm = sysderivatif[-1].subs(
        {alph: alphacoef for alph, alphacoef in zip(alpha, sol)})
    polyadm = polyadm.subs({asymb: aval for asymb, aval in zip(a, avalue)})
    polyadm = sp.simplify(polyadm)
    s0range = np.arange(-10, 0, 0.01)
    taurange = np.arange(0, 10, 0.01)
    return s0range, taurange, polyadm, s, tau, q, avalue, sysderivatif, \
        alpha, a, sol


def compute_root_spectrum_mid_co(n, m, value_tau, a_coeff_mid_co, b_coeff_mid_co):
    
    s = sp.symbols('s')
    tau = sp.symbols('tau')
    a = sp.symbols(["a{:d}".format(i) for i in range(n)], real=True)
    alpha = sp.symbols(["alpha{:d}".format(i) for i in range(m + 1)],
                       real=True)
    polynomial = s ** n + np.array(a).dot([s ** i for i in range(n)])
    delayed = np.array(alpha).dot([s ** i for i in range(m + 1)]) * sp.exp(
        -s * tau)
    q = polynomial + delayed
    sysderivatif = [q]
    for i in range(n + m + 1):
        dernierederivee = sysderivatif[-1]
        sysderivatif.append(dernierederivee.diff(s))
        
    a_num = a_coeff_mid_co.copy()
    a_coeff_mid_co.append(1)
    
    alpha_num = b_coeff_mid_co.copy()
    
    qnumerique = s ** n + np.array(a_num).dot([s ** i for i in range(n)]) + \
        np.array(alpha_num).dot([s ** i for i in range(m + 1)]) * \
        sp.exp(-s * tau)
    qnumerique = qnumerique.subs(tau, value_tau)
    sysrootfinding = [qnumerique, qnumerique.diff(s)]
    sysfunc = [sp.lambdify(s, i) for i in sysrootfinding]
    rect = cx.Rectangle([-100, 10], [-100, 100])
    roots = rect.roots(sysfunc[0], sysfunc[1], rootErrTol=1e-5, absTol=1e-5,
                       M=n + m + 1)
    xroot = np.real(roots[0])
    yroot = np.imag(roots[0])
    return xroot, yroot, qnumerique, a_coeff_mid_co, b_coeff_mid_co


def getroots(m, q, delay, avalue, alphavalue, xwindow, ywindow, s, a, alpha,
             tau):
    derivees = [q, q.diff(s)]
    for i in range(len(derivees)):
        derivees[i] = derivees[i].subs(
            {ai: ai_num for ai, ai_num in zip(a, avalue)})
        derivees[i] = derivees[i].subs(
            {alphai: alphai_num for alphai, alphai_num in
             zip(alpha, alphavalue)})
        derivees[i] = derivees[i].subs({tau: delay})
    func = [sp.lambdify(s, i) for i in derivees]
    rect = cx.Rectangle(xwindow, ywindow)
    roots = rect.roots(func[0], func[1], rootErrTol=1e-5, absTol=1e-5, M=m + 2)
    xroot, yroot = np.real(roots[0]), np.imag(roots[0])
    return xroot, yroot, func


def solve_tau_connu(tau_val, acoef, q, m, s, sysderivatif, alpha, a, tau):
    sys = [q]
    for i in range(m + 1):
        dernierederivee = sys[-1]
        sys.append(dernierederivee.diff(s))
    sol = sp.linsolve(sysderivatif[:-1], alpha).args[0]
    finaleq = sysderivatif[-1].subs(
        {alph: alphacoef for alph, alphacoef in zip(alpha, sol)})
    finaleq = finaleq.subs({asymb: aval for asymb, aval in zip(a, acoef)})
    sols0 = finaleq.subs({tau: tau_val})
    sols0 = sp.solve(sols0)
    sols0eval = [i.evalf() for i in sols0]
    try:
        solution = max([i for i in sols0eval if i < 0])
    except Exception:
        traceback.print_exc()
    return solution


def compute_sensibilite(value_tau, q, m, s, sysderivatif, alpha, a, tau, a_coeff_mid_co, nbit, step):
    tau_nominal = value_tau
    values = sorted([0] + [-step * i for i in range(1, nbit + 1)
                           ] + [step * i for i in range(1, nbit + 1)])
    tau_sens = []
    s0_sens = []
    
    s0_nominal = solve_tau_connu(tau_nominal, a_coeff_mid_co, q, m, s,
                                   sysderivatif, alpha, a, tau)
    
    for value in values:
        tau_sens.append(tau_nominal + value)
        s0_sens.append(solve_tau_connu(tau_sens[-1], a_coeff_mid_co, q, m, s,
                                       sysderivatif, alpha, a, tau))
    normaliser = clr.Normalize(min(tau_sens), max(tau_sens))
    colormapper = cm.ScalarMappable(norm=normaliser,
                                    cmap=plt.get_cmap("rainbow"))
    colormapper.set_array(tau_sens)
    return tau_sens, s0_sens, colormapper, s0_nominal


# -----------------------------------------------------------------------------
# Generic CRRID tab


def compute_crrid_generic_root(n, m, s0, d, value_tau, a_coeff_crrid_generic, b_coeff_mid_co):
    slist = [s0 - i * d for i in range(n + m + 1)]
    s = sp.symbols('s')
    tau = sp.symbols('tau')
    a = sp.symbols(["a{:d}".format(i) for i in range(n)], real=True)
    alpha = sp.symbols(["alpha{:d}".format(i) for i in range(m + 1)],
                       real=True)
    polynomial = s ** n + np.array(a).dot([s ** i for i in range(n)])
    delayed = np.array(alpha).dot([s ** i for i in range(m + 1)]) * sp.exp(
        -s * tau)
    q = polynomial + delayed
    sys = [q] * (n + m + 1)
    for i in range(n + m + 1):
        sys[i] = sys[i].subs({s: slist[i]})
        sys[i] = sys[i].subs({tau: value_tau})
    sol = sp.linsolve(sys, alpha + a).args[0]
    solnum = sol.subs({tau: value_tau})
    a_num = list(solnum[m + 1:])
    a_coeff_crrid_generic = a_num.copy()
    a_coeff_crrid_generic.append(1)
    alpha_num = list(solnum[:m + 1])
    b_coeff_crrid_generic = alpha_num.copy()
    qnumerique = s ** n + np.array(a_num).dot([s ** i for i in range(n)]) + \
        np.array(alpha_num).dot(
                     [s ** i for i in range(m + 1)]) * sp.exp(-s * tau)
    qnumerique = qnumerique.subs(tau, value_tau)
    sysrootfinding = [qnumerique, qnumerique.diff(s)]
    sysfunc = [sp.lambdify(s, i) for i in sysrootfinding]
    rect = cx.Rectangle([-100, 10], [-100, 100])
    roots = rect.roots(sysfunc[0], sysfunc[1], rootErrTol=1e-5, absTol=1e-5,
                       M=n + m + 1)
    xroot = np.real(roots[0])
    yroot = np.imag(roots[0])
    return xroot, yroot, qnumerique, a_coeff_crrid_generic, b_coeff_crrid_generic


# -----------------------------------------------------------------------------
# Time simulation


def explicit_euler(n, tau, acoef, bcoef, t_final, n_iter, init_type,
                   init_args):
    while acoef.size < n:
        acoef = np.append(acoef, 0)
    a0 = np.diag(np.ones(n - 1), 1)
    a0[-1, :] = -1 * acoef
    while bcoef.size < n:
        bcoef = np.append(bcoef, 0)
    a1 = np.zeros((n, n))
    a1[-1, :] = -1 * bcoef
    dt = t_final / (n_iter - 1)
    npast = int(np.ceil(tau / dt))
    dt = tau / npast
    time = np.linspace(start=float(tau), stop=t_final, num=n_iter)
    sol = np.zeros((n, time.size))
    sol[:, :(npast + 1)] = initial_solution(init_type, init_args, n,
                                            time[:(npast + 1)])
    for i in range(npast, time.size - 1):
        sol[:, i + 1] = sol[:, i] + dt * (
                a0.dot(sol[:, i]) + a1.dot(sol[:, i - npast]))
    return time, sol


def initial_solution(init_type, init_args, n, t):
    x = sp.symbols('x')
    if init_type == 'Constant':
        y0 = np.zeros((n, t.size))
        y0[0, :] = init_args
        return y0

    elif init_type == 'Polynomial':

        degree = init_args[0]
        p = np.array(init_args[1:]).dot([x ** i for i in range(degree + 1)])
        derivees = []
        y0 = np.zeros((n, t.size))
        for i in range(n):
            if i == 0:
                derivees.append(p)
            else:
                derivees.append(derivees[-1].diff(x))
            for j in range(t.size):
                y0[i, j] = derivees[i].subs({x: t[j]})
        return y0

    elif init_type == 'Exponential':
        gain = init_args[0]
        exponant = init_args[1]
        e = gain * sp.exp(exponant * x)
        derivees = []
        y0 = np.zeros((n, t.size))
        for i in range(n):
            if i == 0:
                derivees.append(e)
            else:
                derivees.append(derivees[-1].diff(x))
            for j in range(t.size):
                y0[i, j] = derivees[i].subs({x: t[j]})
        return y0

    elif init_type == 'Trigonometric':
        ampl = init_args[0]
        freq = init_args[1]
        phase = init_args[2]
        tt = ampl * sp.sin(freq * x + phase)
        derivees = []
        y0 = np.zeros((n, t.size))
        for i in range(n):
            if i == 0:
                derivees.append(tt)
            else:
                derivees.append(derivees[-1].diff(x))
            for j in range(t.size):
                y0[i, j] = derivees[i].subs({x: t[j]})
        return y0
    else:
        pass


def initialsolution(degn, delay, acoef, bcoef, t_final, n_iter, init_type,
                    init_args):
    acoef = np.array(acoef[:-1])
    bcoef = np.array(bcoef)
    time, sol = explicit_euler(degn, delay, acoef, bcoef, t_final, n_iter,
                               init_type, init_args)
    return time, sol


# -----------------------------------------------------------------------------
# Example 1


def define_symbols(deg):
    """
    :param deg: degré des polynômes
    :return: tous les symboles nécessaires pour la suite
    """
    s = sp.Symbol('s', complex=True)
    n_0 = sp.Symbol("n_0")
    nr_0 = sp.Symbol("nr_0")
    d_0 = sp.Symbol("d_0")
    dr_0 = sp.Symbol("dr_0")
    tau = sp.Symbol('tau', positive=True)
    nuy_sym = sp.symbols(["n_uy{:d}".format(i) for i in range(deg + 1)], real=True)
    a_sym = sp.symbols(["a{:d}".format(i) for i in range(deg)], real=True)
    return s, tau, n_0, nr_0, d_0, dr_0, nuy_sym, a_sym


def construct_polynomials(deg, s, a_sym, nuy_sym):
    """
    :param deg: degré des polynômes psi et nuy (2*nb_modes)
    :param s: symbole s
    :param a_sym: liste regroupant les symboles a{i}
    :param nuy_sym: liste regroupant les symboles nuy{i}
    :return: Les polynômes nuy(s) et psi(s)
    """
    n_uy_poly = np.array(nuy_sym).dot([s**i for i in range(len(nuy_sym))])
    psi_poly = s**deg + np.array(a_sym).dot([s**i for i in range(len(a_sym))])
    return n_uy_poly, psi_poly


def construct_controller_polynomial(s, n_0, nr_0, d_0, dr_0, tau):
    """
    :param s: symbole s (variable complexe)
    :param n_0: symbole n_0
    :param nr_0: ssymbole n_0
    :param d_0: symbole d_0
    :param dr_0: symbole dr_0
    :param tau: symbole du retard réel positif tau
    :return: Les deux polynômes de la fonction de transfert
             du contrôleur
    """
    N_poly = n_0 + nr_0 * sp.exp(-tau * s)
    D_poly = d_0 + dr_0 * sp.exp(-tau * s)
    return N_poly, D_poly


def contruct_quasipolynomial(deg, s, n_0, nr_0, d_0, dr_0, tau, a_sym, nuy_sym):
    """
    :param deg: degré des poly psi(s) et nuy(s)
    :param s: variable complexe s
    :param n_0: symbole n_0
    :param nr_0: symbole nr_0
    :param d_0: symbole d_0
    :param dr_0: symbole dr_0
    :param tau: symbole du retard réel positif tau
    :param a_sym: liste regroupant les symboles a{i}
    :param nuy_sym: liste regroupant les symboles nuy{i}
    :return:
    """
    n_uy_poly, psi_poly = construct_polynomials(deg, s, a_sym, nuy_sym)
    N_poly, D_poly = construct_controller_polynomial(s, n_0, nr_0, d_0, dr_0, tau)
    delta_poly = sp.collect(sp.collect(sp.expand(psi_poly * D_poly - n_uy_poly * N_poly),
                                       sp.exp(-s * tau)), s)
    return delta_poly


def extract_PO_P1(delta, s, tau):
    """
    :param delta: Expression du quasipolynôme
    :param s: variable complexe s
    :param tau: symbole du retard réel positif tau
    :return: Les polynômes P0 et P1 correspondants respectivement
             à la partie non-retardée et à la partie retardée du
             quasipolynôme
    """
    P1 = delta.coeff(sp.exp(-s * tau))
    P0 = delta - sp.exp(-s * tau) * P1
    return P0, P1


def extract_highest_degree_monomial_coeff(P0, P1, deg, s):
    """
    Normalisation du système : on transforme en un système unitaire
    et retardé
    :param P0: Polynôme de la partie non-retardée du quasipolynôme
    :param P1: Polynôme de la partie retardée du quasipolynôme
    :param deg: Degré de psi(s) et nuy(s)
    :param s: variable complexe s
    :return: Le coefficient du monôme de plus haut degré de chaque
             polynôme
    """
    poly1 = P1.coeff(s ** deg)
    poly0 = P0.coeff(s ** deg)
    return poly0, poly1


def solution_n0_nr_0_and_subs(delta, s, tau, n_0, nr_0, deg):
    """
    On élimine les deux variables n0 et nr0 du système, on les
    exprime en fonction de d0 et dr0
    :param delta: Expression du quasipolynôme
    :param s: Variable complexe s
    :param tau: Variable du retard tau
    :return: Les expressions de n0 et nr0 (en fct de d0 et dr0)
             ainsi que le quasipolynôme dans lequel on a
             substitué les expressions de n_0 et nr_0.
    """
    P0, P1 = extract_PO_P1(delta, s, tau)
    poly0, poly1 = extract_highest_degree_monomial_coeff(P0, P1, deg, s)
    ELIM1 = sp.nonlinsolve([poly1, poly0 - 1], [n_0, nr_0])
    n_0_SOL_sym = ELIM1.args[0][0]
    nr_0_SOL_sym = ELIM1.args[0][1]
    delta_post_subs = delta.subs({n_0: n_0_SOL_sym, nr_0: nr_0_SOL_sym})
    return n_0_SOL_sym, nr_0_SOL_sym, delta_post_subs


def solve_d0_dr_0(delta, s, d_0, dr_0):
    """
    Résolution de d_0 et dr_0
    :param delta: Expression du quasipolynôme
    :param s: variable complexe s
    :return: Les expressions de d_0 et dr_0
    """
    syst = [delta, sp.diff(delta, s, 1)]
    ELIM2 = sp.nonlinsolve(syst, [d_0, dr_0])
    d_0_sol = sp.collect(ELIM2.args[0][0], s)
    dr_0_sol = ELIM2.args[0][1]
    return d_0_sol, dr_0_sol


def construct_admissibility_poly(delta, s, tau, d_0_sol, dr_0_sol, d_0, dr_0):
    """
    Détermination du polynôme qui permet d'obtenir la
    courbe d'admissibilité.
    :param delta: Expression du quasipolynôme
    :param s: variable complexe s
    :param tau: variable du retard tau
    :param d_0_sol: Expression de d_0
    :param dr_0_sol: Expression de dr_0
    :return: Polynôme d'admissibilité, il s'agit de la dérivée
             seconde de Delta dans laquelle on a substitué les
             expressions de d_0 et dr_0
    """
    numer_adm, _ = sp.fraction(sp.simplify(sp.diff(delta, s, 2).subs([(d_0, d_0_sol), (dr_0, dr_0_sol)])))
    return numer_adm


def admissibility_plot(poly_admissibility, s, tau, max_tau, min_s0, nb_points=100):
    """
    Tracé du plot d'admissibilité
    :param poly_admissibility: Polynôme pour l'admissibilité
    :param s: variable complexe s
    :param tau: variable du retard positif tau
    :param max_tau: valeur maximale de tau pour le plot
    :param min_s0: valeur minimale de partie réelle pour la dominance s0
                   dans le plot
    :param nb_points: Nombre de points pour le plot
    :return: Plot d'admissibilité
    """
    s_adm_plot = np.linspace(min_s0, 0, nb_points)
    tau_adm_plot = np.linspace(0, max_tau, nb_points)
    s_adm_plot_mesh, tau_adm_plot_mesh = np.meshgrid(s_adm_plot, tau_adm_plot)

    adm_poly_num = sp.lambdify([tau, s], poly_admissibility)
    Z = adm_poly_num(tau_adm_plot_mesh, s_adm_plot_mesh)

    plt.contour(s_adm_plot_mesh, tau_adm_plot_mesh, Z, [0])
    plt.grid()
    plt.show()


def solve_tau(poly_adm, s, tau, s0):
    """
    Détermination de la valeur de tau
    :param poly_adm: Polynôme utilisé pour l'admissibilité.
                     Aussi la dérivée seconde de Delta
    :param s: variable complexe s
    :param tau: variable tau du retard
    :param s0: dominance assignée
    :return: Valeur numérique du retard tau
    """
    tau_sol = sp.solve(poly_adm.subs(s, s0), tau)[0]
    return tau_sol


def subs_parameters(expr, nuy_num, nuy_sym, a_num, a_sym):
    """
    Fonction qui sert à substituer les coefficients du modèle
    pour une expression quelconque
    :param expr: une expression quelconque
    :param nuy_num: liste des coefficients numériques nuy{i}
    :param nuy_sym: liste des symboles nuy{i}
    :param a_num: liste des coefficients numériques a{i}
    :param a_sym: liste des symboles a{i}
    :return: expression avec substitutions des symboles par les
             valeurs numériques correspondantes
    """
    expr = expr.subs({ai: a_numi for ai, a_numi in zip(a_sym, a_num)})
    expr = expr.subs({n_uyi: nuy_numi for n_uyi, nuy_numi in zip(nuy_sym, nuy_num)})
    return expr


def compute_and_display_gains(num_tau, d_0_sol, dr_0_sol, n_0_sol, nr_0_sol, s0,
                              a_sym, a_num, nuy_sym, nuy_num, s, tau, d_0, dr_0):
    """
    Permet de calculer tous les gains à partir des solutions
    symboliques obtenues
    :param num_tau: valeur numérique de tau
    :param d_0_sol: expression symbolique de la solution d_0
    :param dr_0_sol: expression symbolique de la solution d_r_0
    :param n_0_sol: expression symbolique de la solution n_0
    :param nr_0_sol: expression symbolique de la solution nr_0
    :param s0: dominance assignée
    :param a_sym: liste des symboles a{i}
    :param a_num: liste des coefficients numériques a{i}
    :param nuy_sym: liste des symboles nuy{i}
    :param nuy_num: liste des coefficients numériques nuy{i}
    :return: Valeurs numériques de : d_0, dr_0, n_0, nr_0
    """
    num_d_0 = subs_parameters(d_0_sol, nuy_num, nuy_sym,
                              a_num, a_sym).subs(s, s0).subs(tau, num_tau)
    num_dr_0 = subs_parameters(dr_0_sol, nuy_num, nuy_sym,
                               a_num, a_sym).subs(s, s0).subs(tau, num_tau)
    num_n_0 = subs_parameters(n_0_sol, nuy_num, nuy_sym,
                              a_num, a_sym).subs(d_0, num_d_0)
    num_nr_0 = subs_parameters(nr_0_sol, nuy_num, nuy_sym,
                              a_num, a_sym).subs(dr_0, num_dr_0)
    return num_n_0, num_nr_0, num_d_0, num_dr_0, num_tau


def get_final_quasipolynomial(delta, s, tau, num_d_0, num_dr_0, num_tau,
                              a_sym, a_num, nuy_sym, nuy_num, d_0, dr_0):
    """
    Substitutions pour obtenir l'expression finale du quasipolynôme
    qui dépendra donc uniquement de s
    :param delta: expression du quasipolynôme
    :param s: variable symbolique complexe s
    :param tau: variable symbolique du retard
    :param num_d_0: valeur numérique de d_0
    :param num_dr_0: valeur numérique de dr_0
    :param num_tau: valeur numérique de tau
    :param a_sym: liste des symboles a{i}
    :param a_num: liste des coefficients numériques a{i}
    :param nuy_sym: liste des symboles nuy{i}
    :param nuy_num: liste des coefficients numériques nuy{i}
    :return: Expression finale du quasipolynôme
    """
    delta = subs_parameters(delta, nuy_num, nuy_sym, a_num, a_sym)
    substitution = [(dr_0, num_dr_0), (d_0, num_d_0), (tau, num_tau)]
    delta = delta.subs(substitution)
    return delta


def find_roots(delta, s, xwindow=(-100, 10), ywindow=(-200, 200)):
    """
    Recharche des racines complexes du quasipolynôme basée sur le
    principe de l'argument
    :param delta: Expression du quasipolynôme
    :param s: variable symbolique complexe s
    :param xwindow: Intervalle en x (partie réelle) pour
                    le contour de résolution
    :param ywindow: Intervalle en y (partie imaginaire) pour
                    le contour de résolution
    :return: Les valeurs spectrales parmi le contour choisi
    """
    delta_der = sp.diff(delta, s, 1)

    lambda_delta = sp.lambdify(s, delta)
    lambda_delta_der = sp.lambdify(s, delta_der)

    rect = cx.Rectangle(xwindow, ywindow)

    roots = rect.roots(lambda_delta, lambda_delta_der, rootErrTol=1e-5,
                       absTol=1e-5)
    xroot = np.real(roots[0])
    yroot = np.imag(roots[0])
    return xroot, yroot, roots


def plot_spectral_distribution(xroot, yroot):
    """
    Affichage de la distribution spectrale
    :param xroot: parties réelles des valeurs spectrales
    :param yroot: parties imaginaires des valeurs spectrales
    :return: Distribution spectrale
    """
    plt.scatter(xroot, yroot, marker="o", zorder=4)
    plt.axvline(0, linewidth=2, color="black", zorder=3)
    plt.axhline(0, color="black", zorder=2)
    plt.grid(True, zorder=1)
    plt.title("Spectral ditribution")
    plt.xlabel(r'$\Re (s)$')
    plt.ylabel(r'$\Im (s)$')
    plt.show()
    