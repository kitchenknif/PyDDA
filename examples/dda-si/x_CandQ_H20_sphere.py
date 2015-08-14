import time

from numpy import *

from matplotlib.pyplot import *
import scipy.sparse.linalg

from dda_funcs import *
from polarizability_models import *
from misc import *

tic = time.time()
iterative = 1  # 0: none, 1: gmres, 2: minres, 3: qmr

E0 = [1, 1, 0]
# m1 = 1.33 # relative refractive index of water
m1 = 1.33 + .1j  # imag. component to demonstrate absorption
k = 2 * pi  # wave number
d = 1 / (abs(m1) * k)  # lattice spacing

# number of dipoles in the approximate sphere more is required as the
# radius increases
# nrange = [8 32 136 280 552 912 1472 2176 3112 4224 5616 7208 9328 11536]
nrange = array([8, 32, 136, 280, 552, 912, 1472])


# the corresponding effective radii of the spheres
arange = power((3 * nrange / (4 * pi)), (1 / 3)) * d

Cscat = zeros(nrange.size, dtype=numpy.complex128)
Cext = zeros(nrange.size, dtype=numpy.complex128)
Cabs = zeros(nrange.size, dtype=numpy.complex128)

ix = 0  # index, counter
for N in nrange:
    m = m1 * ones([N])
    kvec = [0, 0, k]
    rfile = '../shape/sphere_' + str(nrange[ix]) + '.txt'
    # S = load(rfile)
    S = array(load_dipole_file(rfile))
    r = d * asarray([S[:, 0], S[:, 1], S[:, 2]], dtype=numpy.complex128).T
    Ei = E_inc(E0, kvec, r)
    alph = polarizability_LDR(d, m, kvec, E0)

    A = interaction_A(k, r, alph)

    if iterative == 0:
        P, dummy = linalg.solve(A, Ei)
    elif iterative == 1:
        P, dummy = scipy.sparse.linalg.gmres(A, Ei)
    elif iterative == 2:
        P, dummy = scipy.sparse.linalg.minres(A, Ei)
    elif iterative == 3:
        P, dummy = scipy.sparse.linalg.qmr(A, Ei)

    Cext[ix] = C_ext(k, E0, Ei, P)
    Cabs[ix] = C_abs(k, E0, Ei, P, alph)
    Cscat[ix] = Cext[ix] - Cabs[ix]
    ix += 1

print(time.time() - tic)

# the interaction matrix may take up a lot of memory so it can be cleared
# when no longer required
del A


# Here, we plot the efficiencies Q instead of the cross sections C
# Q = C/(pi*r^2)
clf()
plot(k * arange, Cext / (pi * arange ** 2), '*')
plot(k * arange, Cabs / (pi * arange ** 2), 'x')
plot(k * arange, Cscat / (pi * arange ** 2), 'o')
legend(['Q_{ext}', 'Q_{abs}', 'Q_{scat}'])
ylabel('Q')
xlabel('2\pia/\lambda')  # size parameter
title('m = ' + str(m1))
show(block=True)
