import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

from polarizability_models import *
from dda_funcs import *
from dda_si_funcs import *
from experiment_sim import *

from PyTMM import refractiveIndex
import scatterer
import plot_funcs

pow1d3 = power_function(1. / 3.)
pow2 = power_function(2)
pow3 = power_function(3)

points = 100

# Spherical particle
lambda_range = np.linspace(400, 800, points)  # nm
diameter = 150 # nm

k = 2 * np.pi  # wave number

r, N, d_old = scatterer.dipole_sphere(9, diameter)
plot_funcs.plot_dipoles(r)

catalog = refractiveIndex.RefractiveIndex("../../../../RefractiveIndex/")
Si = catalog.getMaterial('main', 'Si', 'Vuye-20C')

nSi = np.asarray([Si.getRefractiveIndex(lam) + Si.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

I_scat = np.zeros(lambda_range.size)

#
# incident plane wave
#

#Incident field wave vector angle
gamma_deg = 22.5
gamma = gamma_deg / 180 * np.pi
kvec = k * np.asarray([0, np.sin(gamma), -np.cos(gamma)])  # wave vector [x y z]

#Incident field polarization
#E0 = np.asarray([0, cos(gamma), sin(gamma)])  # p-pol
E0 = np.asarray([1, 0, 0])  # E-field [x y z]  # s-pol

#
#
#
for i, lam in enumerate(lambda_range):
    print("Calcualting wavelength {} ".format(lam))
    n = nSi[i]  # refractive index of sphere
    m = n * np.ones([N])

    a_eff = diameter / (2 * lam)  # effective radius in wavelengths
    d_new = pow1d3(4 / 3 * np.pi / N) * a_eff
    r = (d_new/d_old) * r
    d_old = d_new

    Ei = E_inc(E0, kvec, r)  # direct incident field at dipoles
    alph = polarizability_LDR(d_new, m, kvec)  # polarizability of dipoles
    A = interaction_A(k, r, alph)
    P = scipy.sparse.linalg.gmres(A, Ei)[0]

    I_scat[i] = objective_collection(k, r, P, 0.45, 100)

plt.figure(1)
plt.plot(lambda_range, I_scat)
#title(['gamma = ' + str(gamma * 180 / pi) + ', zgap = ' + str(zgap) + ', N = ' + str(N)])
plt.show(block=True)