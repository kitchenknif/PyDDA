# single Ag sphere on BK7 glass
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
import scipy.sparse.linalg
import time

from misc import *
from polarizability_models import *
from dda_funcs import *
from dda_si_funcs import *

import refractiveIndex



pow1d3 = power_function(1. / 3.)
pow2 = power_function(2)

use_mex = 0
step = 20  # lambda step (nm)

# Spherical particle
lambda_range = linspace(280, 600, step)  # nm
diameter = 40  # nm
zgap = 0  # gap btw sphere and substrate (fraction of radius)

gamma_deg = 0
gamma = gamma_deg / 180 * pi
k = 2 * pi  # wave number

r0 = load_dipole_file('../shape/sphere_32.txt')
N = 32
#r0 = load_dipole_file('../shape/sphere_136.txt')
#N = 136
# r0 = dlmread('../../shape/sphere_280.txt'); N = 280;
# r0 = dlmread('../../shape/sphere_552.txt'); N = 552;
# r0 = dlmread('../../shape/sphere_912.txt'); N = 912;
# r0 = dlmread('../../shape/sphere_1472.txt'); N = 1472;



catalog = refractiveIndex.RefractiveIndex()
BK7 = catalog.getMaterial('glass', 'BK7', 'HIKARI')
Ag = catalog.getMaterial('main', 'Ag', 'Rakic')

nBK7 = asarray([BK7.getRefractiveIndex(lam) for lam in lambda_range])
nAg = asarray([Ag.getRefractiveIndex(lam) + Ag.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

Q_abs = zeros(lambda_range.size, dtype=numpy.complex128)
Q_abs_fs = zeros(lambda_range.size, dtype=numpy.complex128)
Q_ext = zeros(lambda_range.size, dtype=numpy.complex128)
Q_ext_fs = zeros(lambda_range.size, dtype=numpy.complex128)
R_p = zeros(lambda_range.size, dtype=numpy.complex128)

# incident plane wave
E0 = asarray([0, cos(gamma), sin(gamma)])  # E-field [x y z]
kvec = k * asarray([0, sin(gamma), -cos(gamma)])  # wave vector [x y z]


start_t = time.time()
ix = 0
for lam in lambda_range:
    n1 = nBK7[ix]
    mu_r = 1

    n2 = 1
    n3 = nAg[ix]  # refractive index of sphere
    k1 = k * n1
    k2 = k * n2  # for top medium

    n_r = n1 / n2
    m = n3 * ones([N]) / n2
    a_eff = diameter / (2 * lam)  # effective radius in wavelengths
    d = pow1d3(4 / 3 * pi / N) * a_eff
    r = d * r0
    r[:, 2] = r[:, 2] + a_eff + zgap * a_eff

    # incident plane wave
    Ei = E_inc(E0, kvec, r)  # direct incident field at dipoles

    # plane wave reflected off substrate
    refl_TE, refl_TM = Fresnel_coeff_n(n_r, abs(gamma))
    E0_r = refl_TM * asarray([0, -cos(gamma), sin(gamma)])  # E-field [x y z]
    kvec_r = k * asarray([0, sin(gamma), cos(gamma)])  # wave vector [x y z]
    Ei_r = E_inc(E0_r, kvec_r, r)  # reflected incident field at dipoles

    alph = polarizability_LDR(d, m, kvec)  # polarizability of dipoles
    # matrix for direct and reflected interactions
    AR = interaction_AR(k1, k2, r, alph)
    P = scipy.sparse.linalg.gmres(AR, Ei + Ei_r)[0]  # solve dipole moments

    Q_abs[ix] = C_abs(k, E0, add(Ei, Ei_r), P, alph) / (pi * pow2(a_eff))

    ix += 1

print(time.time() - start_t)

figure(1)
plot(lambda_range, Q_abs, lambda_range, Q_abs_fs, '--')
title(['gamma = ' + str(gamma * 180 / pi) + ', zgap = ' + str(zgap) + ', N = ' + str(N)])
show(block=True)