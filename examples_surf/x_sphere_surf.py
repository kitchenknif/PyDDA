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

pow1d3 = power_function(1./3.)

# incident field here undergoes total internal reflection in the lower
# half-space (substrate) resulting in an evanescent field in the upper 
# half space (air or vacuum)

check_Ij = 1
E1s = 0        # TE incident field intensity in substrate
E1p = 1        # TM incident field intensity in substrate
theta_1 = pi/4 # incident angle
lam = 632.8 # nm, wavelength of laser
D = 30         # nm, diameter of sphere

# refractive indices
catalog = refractiveIndex.RefractiveIndex()
Si = catalog.getMaterial('main', 'Si', 'Aspnes')
Ag = catalog.getMaterial('main', 'Ag', 'Rakic')

n_subs = Si.getRefractiveIndex(lam) # silicon substrate
n_sph = Ag.getRefractiveIndex(lam)  # Silver particle

k = 2*pi
k_subs = k*n_subs

# load sphere with coordinates of lattice spacing 1
r = load_dipole_file('../shape/sphere_32.txt')
# the lattice spacing may be too big; just a demo

N, col = r.shape                  # N = no. of dipoles
a_eff = .5*D/lam                  # effective radius, relative to wavelength
d = pow1d3(4/3*pi/N)*a_eff        # lattice spacing based on Nd^3 = 4/3 pi r^3
r = d*r                          # rescale sphere to wavelength units
r[:, 2] += a_eff  # sit the sphere on the surface

# # check dipole model
# figure(1)
# clf()
# plot3d(r[:,1]*lam,r[:,2]*lam,r[:,3]*lam,'o','MarkerSize',d*5*lam)
# view(45,45)
# xlabel('x (nm)')
# ylabel('y (nm)')
# zlabel('z (nm)')
# axis([-D/2 D/2 -D/2 D/2 0 D])
#
# # Note that if either E1s or E1p is zero, E2s or E2p correspondingly will be zero
[kvec,E2s,E2p]=evanescent_E(E1s,E1p,theta_1,n_subs,1)
#
# #Ei = E_evanescent(E0, r, 3.5*lambda/200); # incident field at dipoles
# # Incident field at dipoles.
Ei = E_inc(E2s+E2p, kvec, r)
# # polarity of kvec reversed because of the convention Ei = E_0 exp(-ik.r_j)
#
# # polarizability of particle
alph_sph = polarizability_CM(d/lam, n_sph*ones([N]), k)
#
AR = interaction_AR(k_subs, k, r, alph_sph) # formulate interaction matrix
P = scipy.sparse.linalg.gmres(AR, Ei)[0] # solve dipole moments
#
Ej = reshape(Ei - AR*P, [N,3]) # E-field at each dipole
Ij = dot(Ej, Ej, 2)           # field intensity at each dipole
#plot_I(Ij,r,2,d*20*lam)   # plot normalized intensity
#
