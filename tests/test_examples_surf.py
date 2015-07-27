from unittest import TestCase
from dda_si_funcs import *
from dda_funcs import *
from polarizability_models import *
from numpy import *
from misc import *
import scipy.sparse.linalg
import refractiveIndex


class TestDDA_SI_integration(TestCase):
    def test_x_phase_fn_cube(self):
        pow2 = power_function(2)
        pow1d3 = power_function(1. / 3.)

        lambd = 632.8  # nm
        sides = 400  # nm
        gamma = 65 / 180 * pi  # incident angle in radians
        # gamma = 0
        k = 2 * pi  # wave number

        n2 = 1
        n1 = 3.873960000000000 + 0.015720000000000j
        n3 = 0.056372027972028 + 4.272085874125873j

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

        r_exp = (read_data('test_files/cube_surf/r.txt'))

        if not numpy.allclose(r, r_exp, rtol=1e-5):
            print('r fail')
            self.fail()
        else:
            print('r pass')

        d = sides / nl / lambd
        r = d * r

        # incident plane wave
        E0 = [1, 0, 0]  # E-field [x y z]
        kvec = k * asarray([0, -sin(gamma), -cos(gamma)])  # wave vector [x y z]
        Ei = E_inc(E0, kvec, r)  # incident field at dipoles
        Ei_exp = read_data('test_files/cube_surf/Ei.txt').T

        if not numpy.allclose(Ei, Ei_exp, rtol=1e-5):
            print('Ei fail')
            self.fail()
        else:
            print('Ei pass')


        # reflected incident plane wave
        refl_TE, refl_TM = Fresnel_coeff_n(n1, abs(gamma))

        E0_r = refl_TE * asarray([1, 0, 0])  # E-field [x y z]
        kvec_r = k * asarray([0, -sin(gamma), cos(gamma)])  # wave vector [x y z]
        Ei_r = E_inc(E0_r, kvec_r, r)  # reflected field at dipoles
        Ei_r_exp = read_data('test_files/cube_surf/Ei_r.txt').T

        if not numpy.allclose(Ei_r, Ei_r_exp, rtol=1e-5):
            print('Ei_r fail')
            self.fail()
        else:
            print('Ei_r pass')

        alph = polarizability_CM(d, m, k)  # polarizability of dipoles
        alph_exp = read_data('test_files/cube_surf/alph.txt').T

        if not numpy.allclose(alph, alph_exp, rtol=1e-4):
            print('alph fail')
            self.fail()
        else:
            print('alph pass')

        # matrix for direct and reflected interactions
        AR = interaction_AR(k1, k2, r, alph)  # non-global version, 2 copies of AR
        AR_exp = read_data('test_files/cube_surf/AR.txt')

        if not numpy.allclose(AR, AR_exp, rtol=1e-5):
            print('AR fail')
            self.fail()
        else:
            print('AR pass')

        P = scipy.sparse.linalg.qmr(AR, add(Ei, Ei_r))[0]  # solve dipole moments
        P_exp = read_data('test_files/cube_surf/P.txt').T

        if not numpy.allclose(P, P_exp, rtol=1e-3):
            print('P fail')
            self.fail()
        else:
            print('P pass')

        # calculate scattered field as a function of angles
        # parallel to incident plane
        rE = asarray([det_r * ones(pts).T, theta.T, phi_p.T]).T
        Esca = E_sca_SI(k, r, P, rE[:, 0], rE[:, 1], rE[:, 2], n1)
        Esca_p_exp = read_data('test_files/cube_surf/Esca_p.txt')

        if not numpy.allclose(Esca, Esca_p_exp, rtol=1e-3):
            print('Esca_p fail')
            self.fail()
        else:
            print('Esca_p pass')

        E = Esca
        Ip = pow2(k) * (pow2(det_r)).T * asarray([dot(a, a) for a in E])  # dot(E,E,2)


        # perpendicular to incident plane
        rE = asarray([det_r * ones(pts).T, theta.T, phi_s.T]).T
        Esca = E_sca_SI(k, r, P, rE[:, 0], rE[:, 1], rE[:, 2], n1)
        Esca_p_exp = read_data('test_files/cube_surf/Esca_s.txt')

        if not numpy.allclose(Esca, Esca_p_exp, rtol=1e-3):
            print('Esca_s fail')
            self.fail()
        else:
            print('Esca_s pass')
        E = Esca
        Is = pow2(k) * (pow2(det_r)).T * asarray([dot(a, a) for a in E])  # dot(E,E,2)

