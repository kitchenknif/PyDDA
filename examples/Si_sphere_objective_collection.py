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

import refractiveIndex



pow1d3 = power_function(1. / 3.)
pow2 = power_function(2)

step = 200  # lambda step (nm)

# Spherical particle
lambda_range = linspace(300, 800, step)  # nm
diameter = 150  # nm
zgap = 0  # gap btw sphere and substrate (fraction of radius)

k = 2 * pi  # wave number

#r0 = load_dipole_file('../shape/sphere_32.txt')
#N = 32
r0 = load_dipole_file('../shape/sphere_136.txt')
N = 136
# r0 = dlmread('../../shape/sphere_280.txt'); N = 280;
# r0 = dlmread('../../shape/sphere_552.txt'); N = 552;
# r0 = dlmread('../../shape/sphere_912.txt'); N = 912;
# r0 = dlmread('../../shape/sphere_1472.txt'); N = 1472;


catalog = refractiveIndex.RefractiveIndex()
#BK7 = catalog.getMaterial('glass', 'BK7', 'HIKARI')
Si = catalog.getMaterial('main', 'Si', 'Aspnes')

#nBK7 = asarray([BK7.getRefractiveIndex(lam) for lam in lambda_range])
nSi = asarray([Si.getRefractiveIndex(lam) + Si.getExtinctionCoefficient(lam)*1j for lam in lambda_range])

I_scat = zeros(lambda_range.size)


#
# incident plane wave
#

#Incident field wave vector angle
gamma_deg = 22.5
gamma = gamma_deg / 180 * pi
kvec = k * asarray([0, sin(gamma), -cos(gamma)])  # wave vector [x y z]

#Incident field polarization
E0 = asarray([0, cos(gamma), sin(gamma)])  # p-pol
#E0 = asarray([1, 0, 0])  # E-field [x y z]  # s-pol


#
#
#


ix = 0
for lam in lambda_range:
    n = nSi[ix]  # refractive index of sphere

    m = n * ones([N])
    a_eff = diameter / (2 * lam)  # effective radius in wavelengths
    d = pow1d3(4 / 3 * pi / N) * a_eff
    r = d * r0
    #r[:, 2] = r[:, 2] + a_eff + zgap * a_eff

    # incident plane wave
    Ei = E_inc(E0, kvec, r)  # direct incident field at dipoles

    # plane wave reflected off substrate
    #refl_TE, refl_TM = Fresnel_coeff_n(n_r, abs(gamma))
    #E0_r = refl_TM * asarray([0, -cos(gamma), sin(gamma)])  # E-field [x y z]
    #kvec_r = k * asarray([0, sin(gamma), cos(gamma)])  # wave vector [x y z]
    #Ei_r = E_inc(E0_r, kvec_r, r)  # reflected incident field at dipoles

    alph = polarizability_LDR(d, m, kvec)  # polarizability of dipoles
    # matrix for direct and reflected interactions
    #AR = interaction_AR(k1, k2, r, alph)
    #P = scipy.sparse.linalg.gmres(AR, Ei + Ei_r)[0]  # solve dipole moments

    #Q_abs[ix] = C_abs(k, E0, add(Ei, Ei_r), P, alph) / (pi * pow2(a_eff))

    A = interaction_A(k, r, alph)
    P = scipy.sparse.linalg.gmres(A, Ei)[0]

    I_scat[ix] = objective_collection(k, r, P, 0.45, 100)

    ix += 1


#write_data('Si_150nm_no_surf_p_pol.txt', lambda_range, I_scat)

figure(1)
plot(lambda_range, I_scat)
#title(['gamma = ' + str(gamma * 180 / pi) + ', zgap = ' + str(zgap) + ', N = ' + str(N)])
show(block=True)