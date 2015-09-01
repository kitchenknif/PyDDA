import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

from polarizability_models import *
from dda_funcs import *

from PyTMM import refractiveIndex
import scatterer
import plot_funcs
import spec

from pymiecoated import Mie

pow1d3 = power_function(1. / 3.)
pow2 = power_function(2)
pow3 = power_function(3)

points = 50
# Spherical particle
lambda_range = np.linspace(400, 650, points)  # nm
diameter = 63*2 # nm

k = 2 * np.pi  # wave number

r, N, d_old = scatterer.dipole_sphere(11, diameter)
plot_funcs.plot_dipoles(r)

catalog = refractiveIndex.RefractiveIndex("../../../RefractiveIndex/")
Si = catalog.getMaterial('main', 'Si', 'Vuye-20C')

nSi = np.asarray([Si.getRefractiveIndex(lam) + Si.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

Cext = np.zeros(lambda_range.size)
Cext_mie = np.zeros(lambda_range.size)
#
# incident plane wave
#
kvec = k * np.asarray([0, 0, 1])  # wave vector [x y z]
E0 = np.asarray([1, 0, 0])  # Incident field polarization
#
#
#

for ix, lam in enumerate(lambda_range):
    print("Calcualting wavelength {} ".format(lam))
    n = nSi[ix]  # refractive index of sphere

    m = n * np.ones([N])

    a_eff = diameter / (2 * lam)  # effective radius in wavelengths

    d_new = pow1d3(4 / 3 * np.pi / N) * a_eff
    #d_new = a_eff / 10
    r *= (d_new / d_old)
    d_old = d_new

    # incident plane wave
    Ei = E_inc(E0, kvec, r)  # direct incident field at dipoles

    alph = polarizability_LDR(d_new, m, kvec)  # polarizability of dipoles

    A = interaction_A(k, r, alph)
    P = scipy.sparse.linalg.gmres(A, Ei)[0]

    Cext[ix] = C_ext(k, E0, Ei, P) * (lam/100)**2 / 100  # Convert to um**2

    mie = Mie(x=2*np.pi*a_eff, m=n)
    Cext_mie[ix] = mie.qext() * np.pi * (a_eff/2)**2  # Convert cross section
    del(A)


evlukhin = spec.Spec.loadSpecFromASCII("./reference/sphere_r63nm.txt", ',')

#Cext = numpy.divide(Cext, numpy.max(Cext))
#Cext_mie = numpy.divide(Cext_mie, numpy.max(Cext_mie))
#evlukhin.data = numpy.divide(evlukhin.data, numpy.max(evlukhin.data))


plt.figure(1)
plt.plot(lambda_range, Cext)
plt.plot(evlukhin.wavelengths, evlukhin.data)
plt.plot(lambda_range, Cext_mie)
plt.ylabel("$\sigma_{ext}$,  $\mu$m$^2$")
plt.xlabel("wavelength,  nm")
plt.legend(['PyDDA', 'Evlukhin', 'Mie theory'])
plt.show(block=True)
