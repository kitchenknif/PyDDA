# single Ag sphere on BK7 glass
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

from polarizability_models import *
from dda_funcs import *
from dda_si_funcs import *
from experiment_sim import *
import plot_funcs

from PyTMM import refractiveIndex
import scatterer


pow1d3 = power_function(1. / 3.)
pow2 = power_function(2)

points = 50  # points

# Spherical particle
lambda_range = np.linspace(450, 820, points)  # nm
diameter = 97*2  # nm
zgap = 0  # gap btw sphere and substrate (fraction of radius)


k = 2 * np.pi  # wave number


r0, N, d_old = scatterer.dipole_sphere(7, diameter)
r0 /= d_old
#plot_funcs.plot_dipoles(r0)

catalog = refractiveIndex.RefractiveIndex("../../../../RefractiveIndex/")
BK7 = catalog.getMaterial('glass', 'BK7', 'HIKARI')
Si = catalog.getMaterial('main', 'Si', 'Aspnes')

nBK7 = np.asarray([BK7.getRefractiveIndex(lam) for lam in lambda_range])
nSi = np.asarray([Si.getRefractiveIndex(lam) + Si.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

I_scat = np.zeros(lambda_range.size)

#
# incident plane wave
#

# Incident field polarization
E0 = np.asarray([1, 0, 0])  # E-field [x y z]

# Incident field wave vector angle
gamma_deg = 22.5
gamma = gamma_deg / 180 * np.pi
kvec = k * np.asarray([0, np.sin(gamma), -np.cos(gamma)])  # wave vector [x y z]


for ix, lam in enumerate(lambda_range):
    print("Calcualting wavelength {}".format(lam))

    n1 = nBK7[ix]
    mu_r = 1

    n2 = 1
    n3 = nSi[ix]  # refractive index of sphere
    k1 = k * n1
    k2 = k * n2  # for top medium

    n_r = n1 / n2
    m = n3 * np.ones([N]) / n2
    a_eff = diameter / (2 * lam)  # effective radius in wavelengths
    d = pow1d3(4 / 3 * np.pi / N) * a_eff
    r = d * r0
    r[:, 2] = r[:, 2] + a_eff + zgap * a_eff

    # incident plane wave
    Ei = E_inc(E0, kvec, r)  # direct incident field at dipoles

    # plane wave reflected off substrate
    refl_TE, refl_TM = Fresnel_coeff_n(n_r, abs(gamma))
    E0_r = refl_TM * np.asarray([0, -np.cos(gamma), np.sin(gamma)])  # E-field [x y z]
    kvec_r = k * np.asarray([0, np.sin(gamma), np.cos(gamma)])  # wave vector [x y z]
    Ei_r = E_inc(E0_r, kvec_r, r)  # reflected incident field at dipoles

    alph = polarizability_LDR(d, m, kvec)  # polarizability of dipoles
    # matrix for direct and reflected interactions
    print("Building interaction matrix")
    AR = interaction_AR(k1, k2, r, alph)
    print("Solving")
    P = scipy.sparse.linalg.gmres(AR, Ei + Ei_r)[0]  # solve dipole moments

    I_scat[ix] = objective_collection_si(k, r, P, n1, 0.45, 100)


# plt.figure(1)
# plt.plot(lambda_range, I_scat)
# title(['gamma = ' + str(gamma * 180 / pi) + ', zgap = ' + str(zgap) + ', N = ' + str(N)])
# plt.show(block=True)