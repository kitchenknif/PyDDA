import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

from polarizability_models import *
from dda_funcs import *

from PyTMM import refractiveIndex
import scatterer
import plot_funcs
import spec
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
r, N, d_old = scatterer.dipole_cube(5, diameter)
#plot_funcs.plot_dipoles(r)

points = 60
lambda_range = np.linspace(540, 820, points)  # nm

catalog = refractiveIndex.RefractiveIndex("../../../RefractiveIndex/")
Si = catalog.getMaterial('main', 'Si', 'Vuye-20C')
nSi = np.asarray([Si.getRefractiveIndex(lam) + Si.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

#
# incident plane wave
#
k = 2 * np.pi  # wave number
kvec = k * np.asarray([0, 0, 1])  # wave vector [x y z]
E0 = np.asarray([1, 0, 0])  # Incident field polarization
#
#
#

Cext = np.zeros(lambda_range.size)

for ix, lam in enumerate(lambda_range):
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

    Cext[ix] = C_ext(k, E0, Ei, P) #* (lam/100)**2 / 100 #Convert to um**2

evlukhin = spec.Spec.loadSpecFromASCII("./reference/cube_a160nm.txt", ',')

Cext = numpy.divide(Cext, numpy.max(Cext))
evlukhin.data = numpy.divide(evlukhin.data, numpy.max(evlukhin.data))


plt.figure(1)
plt.plot(lambda_range, Cext)
plt.plot(evlukhin.wavelengths, evlukhin.data)
plt.ylabel("$\sigma_{ext}, \mu m^2$")
plt.xlabel("$wavelength, nm$")
plt.legend(['PyDDA', 'Evlukhin'])
plt.show(block=True)
