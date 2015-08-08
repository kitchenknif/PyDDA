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
from experiment_sim import *

from PyTMM import refractiveIndex
import scatterer


pow1d3 = power_function(1. / 3.)
pow2 = power_function(2)
pow3 = power_function(3)

points = 60

# Spherical particle
lambda_range = linspace(400, 650, points)  # nm
diameter = 63*2 # nm

k = 2 * pi  # wave number

r, N, d_old = scatterer.dipole_sphere(8, 1)
#r = misc.load_dipole_file('../shape/sphere_912.txt')
#N = numpy.shape(r)[0]
#d_old = 1

catalog = refractiveIndex.RefractiveIndex("../../RefractiveIndex/")
Si = catalog.getMaterial('main', 'Si', 'Vuye-20C')

nSi = asarray([Si.getRefractiveIndex(lam) + Si.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

Cscat = zeros(lambda_range.size)
Cext = zeros(lambda_range.size)
Cabs = zeros(lambda_range.size)


#
# incident plane wave
#

#Incident field wave vector angle
gamma_deg = 22.5
gamma = gamma_deg / 180 * pi
kvec = k * asarray([0, 0, 1])  # wave vector [x y z]

#Incident field polarization
E0 = asarray([1, 0, 0])

#
#
#


ix = 0
for lam in lambda_range:
    print("Calcualting wavelength {} ".format(lam))
    n = nSi[ix]  # refractive index of sphere

    m = n * ones([N])

    a_eff = diameter / (2 * lam)  # effective radius in wavelengths

    d_new = pow1d3(4 / 3 * pi / N) * a_eff
    r = (d_new/d_old) * r
    d_old = d_new

    #scat.rescale(1)
    #s = scatterer.rescale_scatterer(scat, a_eff)
    #d = s.dipole_spacing
    #r = s.dipoles

    #d = pow1d3((4/3)*numpy.pi*pow3(a_eff)/N)
    #r = r0 * a_eff

    # incident plane wave
    Ei = E_inc(E0, kvec, r)  # direct incident field at dipoles

    alph = polarizability_LDR(d_new, m, kvec)  # polarizability of dipoles

    A = interaction_A(k, r, alph)
    P = scipy.sparse.linalg.gmres(A, Ei)[0]

    Cext[ix] = C_ext(k, E0, Ei, P)
    Cabs[ix] = C_abs(k, E0, Ei, P, alph)
    Cscat[ix] = Cext[ix] - Cabs[ix]
    ix += 1


#write_data('Si_150nm_no_surf_p_pol.txt', lambda_range, I_scat)

figure(1)
plot(lambda_range, Cscat)
plot(lambda_range, Cabs)
plot(lambda_range, Cext)
legend(['Scat', 'Abs', 'Ext'])
#title(['gamma = ' + str(gamma * 180 / pi) + ', zgap = ' + str(zgap) + ', N = ' + str(N)])
show(block=True)