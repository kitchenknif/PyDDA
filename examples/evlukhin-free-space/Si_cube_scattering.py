import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

from polarizability_models import *
from dda_funcs import *

from PyTMM import refractiveIndex
import scatterer
import plot_funcs
#
#
#

pow1d3 = power_function(1. / 3.)
pow2 = power_function(2)
pow3 = power_function(3)

#
# Particle
#
diameter = 160 # nm
r, N, d_old = scatterer.dipole_cube(7, diameter)
#plot_funcs.plot_dipoles(r)

points = 60
lambda_range = np.linspace(500, 800, points)  # nm

catalog = refractiveIndex.RefractiveIndex("../../../../RefractiveIndex/")
Si = catalog.getMaterial('main', 'Si', 'Vuye-20C')
nSi = np.asarray([Si.getRefractiveIndex(lam) + Si.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

#
# incident plane wave
#

#Incident field wave vector angle
gamma_deg = 22.5
gamma = gamma_deg / 180 * np.pi
k = 2 * np.pi  # wave number
kvec = k * np.asarray([0, 0, 1])  # wave vector [x y z]

#Incident field polarization
E0 = np.asarray([1, 0, 0])

#
#
#

Cscat = np.zeros(lambda_range.size)
Cext = np.zeros(lambda_range.size)
Cabs = np.zeros(lambda_range.size)


ix = 0
for lam in lambda_range:
    print("Calcualting wavelength {} ".format(lam))
    n = nSi[ix]  # refractive index of sphere

    m = n * np.ones([N])

    a_eff = diameter / lam  # effective radius in wavelengths

    d_new = a_eff / pow1d3(N)
    r = (d_new/d_old) * r
    d_old = d_new

    # incident plane wave
    Ei = E_inc(E0, kvec, r)  # direct incident field at dipoles

    alph = polarizability_LDR(d_new, m, kvec)  # polarizability of dipoles

    A = interaction_A(k, r, alph)
    P = scipy.sparse.linalg.gmres(A, Ei)[0]

    Cext[ix] = C_ext(k, E0, Ei, P)
    Cabs[ix] = C_abs(k, E0, Ei, P, alph)
    Cscat[ix] = Cext[ix] - Cabs[ix]
    ix += 1

plt.figure(1)
plt.plot(lambda_range, Cscat)
plt.plot(lambda_range, Cabs)
plt.plot(lambda_range, Cext)
plt.legend(['Scat', 'Abs', 'Ext'])
plt.show(block=True)