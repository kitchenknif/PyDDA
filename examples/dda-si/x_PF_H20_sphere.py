# phase function demo for a water sphere

from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
import scipy.sparse.linalg
import time

from misc import *
from polarizability_models import *
from dda_funcs import *


tic = time.time()

E0 = [1, 0, 0]  # x-polarization
m1 = 1.33
shapepath = '../shape/'
rfile = shapepath + 'sphere_912.txt'

S = load_dipole_file(rfile)

N = S[:, 1].size

k = 2*pi  # wave number
d = 1/(abs(m1)*k)
a_eff = (3*N/(4*pi))**(1/3)*1/(k*abs(m1))
r = d*asarray([S[:, 0], S[:, 1], S[:, 2]], dtype=numpy.complex128).T
m = m1*ones(N)

kvec = [0, 0, k]  # propagating in +z direction
Ei = E_inc(E0, kvec, r)
alph = polarizability_LDR(d, m, k)
A = interaction_A(k, r, alph)
P, dummy = scipy.sparse.linalg.gmres(A, Ei)

rang = linspace(0, pi, 100)

Esca_S = zeros([rang.size])
Esca_P = zeros([rang.size])
Einc_S = zeros([rang.size])
Einc_P = zeros([rang.size])

for ix, theta in enumerate(rang):
  phi = 90  # perpendicular to x-z plane
  r_E = zeros([3])  # evaluation point
  r_E[0], r_E[1], r_E[2] = rtp2xyz(100, theta, phi)
  E = E_sca_FF(k, r, P, r_E)
  Esca_S[ix] = norm(E)
  kr = dot([k, k, k], r_E)
  expikr = exp(multiply(1j, kr))
  E1 = [E0[0]*expikr, E0[1]*expikr, E0[2]*expikr]
  Einc_S[ix] = norm(E1)


  phi = 0
  r_E = zeros([3])  # evaluation point
  r_E[0], r_E[1], r_E[2] = rtp2xyz(100, theta, phi)
  E = E_sca_FF(k, r, P, r_E)
  Esca_P[ix] = norm(E)
  kr = dot([k, k, k], r_E)
  expikr = exp(multiply(1j, kr))
  E1 = [E0[0]*expikr, E0[1]*expikr, E0[2]*expikr]
  Einc_P[ix] = norm(E1)


print(time.time() - tic)

ph = semilogy(rang * 180 / pi, Esca_P**2./Einc_P**2, '--', rang * 180 / pi, Esca_S**2./Einc_S**2)
xlim([0, 180])
ylabel('log |E_{sca}|^2')
xlabel('phase angle')
title('ka = ' + str(k*a_eff) + ', m = ' + str(m1) + ', N = ' + str(N))
legend(['parallel', 'perpendicular'])
show(block=True)

