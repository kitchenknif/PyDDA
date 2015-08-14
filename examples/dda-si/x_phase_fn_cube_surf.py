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

# Ag cube on Si substrate

pow2 = power_function(2)
pow1d3 = power_function(1. / 3.)

lambd = 632.8  # nm

catalog = refractiveIndex.RefractiveIndex()
Si = catalog.getMaterial('main', 'Si', 'Aspnes')
Ag = catalog.getMaterial('main', 'Ag', 'Rakic')

nSi = Si.getRefractiveIndex(lambd) + Si.getExtinctionCoefficient(lambd) * 1j
nAg = Ag.getRefractiveIndex(lambd) + Ag.getExtinctionCoefficient(lambd) * 1j

check_shape = 1

sides = 400  # nm

gamma = 65 / 180 * pi  # incident angle in radians
# gamma = 0
k = 2 * pi  # wave number

# n1 = nSi  # refractive index of Si substrate
n2 = 1
# n3 = nAg  # refractive index Ag cube
n1 = 3.8740 + 0.0157j
n3 = 0.0564 + 4.2721j

k1 = k * n1  # for bottom medium, subtrate
k2 = k * n2  # for top medium

N = 64  # number of dipoles
# N = 1000; # number of dipoles

theta = linspace(-pi / 2, pi / 2, 180)  # phase angle range
pts = theta.size
phi_p = zeros(pts)
phi_s = pi / 2 * ones(pts)
det_r = 100

r = load_dipole_file('../shape/cube_' + str(N) + '.txt')
m = n3 * ones(N)
nl = pow1d3(N)
r[:, 2] += + nl / 2
d = sides / nl / lambd
r *= d

# if check_shape:
#    r_nm = r*lambda;
#    figure(1)
#    clf
#    hold on
#    plot3(r_nm(:,1),r_nm(:,2),r_nm(:,3),'o','MarkerSize',d*lambda*.59)
#    axis([-sides/2 sides/2 -sides/2 sides/2 0 sides]*1.1)
#    axis equal
#    xlabel('x')
#    ylabel('y')
#    view(45,45);
#    hold off
# end

# incident plane wave
E0 = [1, 0, 0]  # E-field [x y z]
kvec = k * asarray([0, -sin(gamma), -cos(gamma)])  # wave vector [x y z]
Ei = E_inc(E0, kvec, r)  # incident field at dipoles

# reflected incident plane wave
refl_TE, refl_TM = Fresnel_coeff_n(n1, abs(gamma))

E0_r = refl_TE * asarray([1, 0, 0])  # E-field [x y z]
kvec_r = k * asarray([0, -sin(gamma), cos(gamma)])  # wave vector [x y z]
Ei_r = E_inc(E0_r, kvec_r, r)  # reflected field at dipoles

alph = polarizability_CM(d, m)  # polarizability of dipoles

# matrix for direct and reflected interactions
AR = interaction_AR(k1, k2, r, alph)  # non-global version, 2 copies of AR


#
#
#
# P = linalg.solve(AR, add(Ei, Ei_r))
# P = scipy.sparse.linalg.gmres(AR, add(Ei, Ei_r))[0]
# P = scipy.sparse.linalg.minres(AR, add(Ei, Ei_r))[0]
P = scipy.sparse.linalg.qmr(AR, add(Ei, Ei_r))[0]  # solve dipole moments
# P = read_data('../tests/test_files/cube_surf/P.txt').T[0]

#
#
#

# calculate scattered field as a function of angles
# parallel to incident plane
rE = asarray([det_r * ones(pts).T, theta.T, phi_p.T]).T
Esca = E_sca_SI(k, r, P, rE[:, 0], rE[:, 1], rE[:, 2], n1)
E = Esca
Ip = pow2(k) * (pow2(det_r)).T * asarray([dot(a.conj(), a) for a in E])  # dot(E,E,2)


# perpendicular to incident plane
rE = asarray([det_r * ones(pts).T, theta.T, phi_s.T]).T
Esca = E_sca_SI(k, r, P, rE[:, 0], rE[:, 1], rE[:, 2], n1)
E = Esca
Is = pow2(k) * (pow2(det_r)).T * asarray([dot(a.conj(), a) for a in E])  # dot(E,E,2)
# 
# #save(['data/I_PF_cube_N' int2str(N) '.mat'],'Ip','Is')
# 
th = theta.T * 180 / pi

semilogy(th, Is, '-', th, Ip, '--')
ylabel('I_{sca}')
xlabel('Scattering Angle')
xlim([-90, 90])
ylim([1e-5, 1])

h = legend(['p', 's'])
# set(h,'Location','SouthEast')
title('N=' + str(N))
# print('-dpng',['data/PF_cube_N' int2str(N) '.png'])

show(block=True)
