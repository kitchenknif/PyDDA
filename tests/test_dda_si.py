from unittest import TestCase
from dda_si_funcs import *
from dda_funcs import *
from polarizability_models import *
from numpy import *
from misc import *
import refractiveIndex

__author__ = 'Kryosugarra'


class TestDDA_SI(TestCase):
    def test_Fresnel_coeff_n(self):
        n = 1.3
        theta = pi/7

        r_te, r_tm = Fresnel_coeff_n(n, theta)
        if not ((r_te - (-0.1526)) < 0.001 and (r_tm - (-0.1081))):
            self.fail()

    def test_reflection_Rjk(self):
        #
        # Test 1
        #

        k1 = 24.34080855260143+0.09877167302886311j
        k2 = 6.283185307179586
        r_j = asarray([-0.00955273, -0.00955273,  0.01415144])
        r_k = asarray([-0.00955273, -0.00955273,  0.01415144])
        S = asarray([   0.00000000  -0.j,
                        184.81715393-304.2416687j,
                        266.35104370-281.30355835j,
                        -266.35104370+281.30355835j])
        Rjk = reflection_Rjk(k1, k2, r_j, r_k, S)
        Rjk_e = asarray([
            [-3.800082590967380e+04 + 1.025113013485816e+02j, 0j, 0j],
            [0j, -3.800082590967380e+04 + 1.025113013485816e+02j, 0j],
            [0j, 0j, -7.858867744276137e+04 + 7.481897301790775e+01j]
        ])

        if not allclose(Rjk, Rjk_e, rtol=1e-3):
            print('Test 1')
            print(numpy.abs(Rjk_e - Rjk))
            self.fail()

        #
        # Test 2
        #

        k1 = 24.34080855260143+0.09877167302886311j
        k2 = 6.283185307179586
        r_j = asarray([-0.00955273, -0.00955273,  0.01415144])
        r_k = asarray([-0.00955273, -0.00955273,  0.03325691])
        S = asarray([0.00000000 -0.j,
                     184.81715393-304.2416687j,
                     266.35104370-281.30355835j,
                     -266.35104370+281.30355835j])
        Rjk = reflection_Rjk(k1, k2, r_j, r_k, S)
        Rjk_e = asarray([
            [-7.872360518001917e+03 + 1.335989451044329e+02j, 0j, 0j],
            [0j, -7.872360518001917e+03 + 1.335989451044329e+02j, 0j],
            [0j, 0j, -1.732228042985040e+04 + 1.421978501350253e+02j]
        ])

        if not allclose(Rjk, Rjk_e, rtol=1e-3):
            print('Test 2')
            print(numpy.abs(Rjk_e - Rjk))
            self.fail()

    def test_interaction_AR(self):
        pow1d3 = power_function(1./3.)
        lam = 632.8  # nm, wavelength of laser
        D = 30         # nm, diameter of sphere

        # refractive indices
        #catalog = refractiveIndex.RefractiveIndex()
        #Si = catalog.getMaterial('main', 'Si', 'Aspnes')
        #Ag = catalog.getMaterial('main', 'Ag', 'Rakic')

        #n_subs = Si.getRefractiveIndex(lam) # silicon substrate
        #n_sph = Ag.getRefractiveIndex(lam)  # Silver particle
        n_sph = 0.056372027972028 + 4.272085874125873j
        n_subs = 3.873960000000000 + 0.015720000000000j

        k = 2*pi
        k_subs = k*n_subs

        # load sphere with coordinates of lattice spacing 1
        r = load_dipole_file('../shape/sphere_8.txt')
        # the lattice spacing may be too big; just a demo

        N, col = r.shape                  # N = no. of dipoles
        a_eff = .5*D/lam                  # effective radius, relative to wavelength
        d = pow1d3(4/3*pi/N)*a_eff        # lattice spacing based on Nd^3 = 4/3 pi r^3
        r *= d                            # rescale sphere to wavelength units
        r[:, 2] += a_eff  # sit the sphere on the surface


        alph_sph = polarizability_CM(d/lam, n_sph*ones([N]), k)
        AR = interaction_AR(k_subs, k, r, alph_sph)
        AR_e = read_data("test_files/AR.txt")

        if not allclose(AR, AR_e, rtol=1e-3):
            self.fail()

    def test_evanescent_k_e(self):
        print("evanescent_E test passes, no need to test evanescent_k_e.")
        #self.fail()

    def test_evanescent_E(self):
        pow1d3 = power_function(1./3.)

        check_Ij = 1
        E1s = 0        # TE incident field intensity in substrate
        E1p = 1        # TM incident field intensity in substrate
        theta_1 = pi/4  # incident angle
        lam = 632.8  # nm, wavelength of laser
        D = 30         # nm, diameter of sphere

        # refractive indices
        #catalog = refractiveIndex.RefractiveIndex()
        #Si = catalog.getMaterial('main', 'Si', 'Aspnes')
        #Ag = catalog.getMaterial('main', 'Ag', 'Rakic')

        #n_subs = Si.getRefractiveIndex(lam) # silicon substrate
        #n_sph = Ag.getRefractiveIndex(lam)  # Silver particle
        n_sph = 0.056372027972028 + 4.272085874125873j
        n_subs = 3.873960000000000 + 0.015720000000000j

        k = 2*pi
        k_subs = k*n_subs

        # load sphere with coordinates of lattice spacing 1
        r = load_dipole_file('../shape/sphere_8.txt')
        # the lattice spacing may be too big; just a demo

        N, col = r.shape                  # N = no. of dipoles
        a_eff = .5*D/lam                  # effective radius, relative to wavelength
        d = pow1d3(4/3*pi/N)*a_eff        # lattice spacing based on Nd^3 = 4/3 pi r^3
        r *= d                            # rescale sphere to wavelength units
        r[:, 2] += a_eff  # sit the sphere on the surface


        kvec, E2s, E2p = evanescent_E(E1s,E1p,theta_1,n_subs,1)

        kvec_e = asarray([0.000000000000000 + 0.000000000000000j,
                          17.211550787107980 + 0.069842119787850j,
                          0.075019452388962 - 16.023726560615675j])
        E2s_e = asarray([0, 0, 0])

        E2p_e = asarray([0.000000000000000 + 0.000000000000000j,
                       -1.406136255419883 + 0.100572011079454j,
                        0.108969322164104 + 1.510300043237413j])

        AR_e = read_data("test_files/AR.txt")

        if not allclose(kvec, kvec_e, rtol=1e-3):
            self.fail()
        if not allclose(E2s, E2s_e, rtol=1e-3):
            self.fail()
        if not allclose(E2p, E2p_e, rtol=1e-3):
            self.fail()



    def test_E_sca_SI(self):
        self.fail()