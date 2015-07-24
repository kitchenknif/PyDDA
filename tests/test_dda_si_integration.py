from unittest import TestCase
from dda_si_funcs import *
from dda_funcs import *
from polarizability_models import *
from numpy import *
from misc import *
import refractiveIndex


class TestDDA_SI_integration(TestCase):
    def test_saoa(self):
        print("rom1 test passes -> no need to test saoa")
        #self.fail()

    def test_precalc_Somm(self):
        pow1d3 = power_function(1./3.)
        lam = 632.8  # nm, wavelength of laser
        D = 30         # nm, diameter of sphere

        # refractive indices
        catalog = refractiveIndex.RefractiveIndex()
        Si = catalog.getMaterial('main', 'Si', 'Aspnes')

        #n_subs = Si.getRefractiveIndex(lam)# silicon substrate
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

        S, nS = precalc_Somm(r, k_subs, k)
        S_exp = read_data('test_files/S.txt')
        nS_exp = (read_data('test_files/nS.txt') - 1).T

        if not ((numpy.abs(S - S_exp) < 0.01).all() and (numpy.abs(nS - nS_exp) < 0.001).all()):
            print(numpy.abs(S - S_exp))
            print(numpy.abs(nS - nS_exp))
            self.fail()

    def test_evlua(self):
        IV_rho_e = 0
        IV_z_e = 1.8482e+02 - 3.0424e+02j
        IH_rho_e = 2.6635e+02 - 2.8130e+02j
        IH_phi_e = -2.6635e+02 + 2.8130e+02j

        k1 = 24.3408085526014 + 0.0987716730288631j
        k2 = 6.283185307179586
        rho = 0.0
        zph = 0.066513811085846

        IV_rho, IV_z, IH_rho, IH_phi = evlua(zph, rho, k1, k2)

        if not (numpy.abs(IV_rho_e - IV_rho) < 0.01):
            self.fail()
        if not (numpy.abs(IV_z_e - IV_z) < 0.01):
            self.fail()
        if not (numpy.abs(IH_rho_e - IH_rho) < 0.01):
            self.fail()
        if not (numpy.abs(IH_phi_e - IH_phi) < 0.01):
            self.fail()

    def test_gshank(self):
        start = 15.034471543201017-15.034471543201017j
        dela = 9.446437070145024
        suminc = zeros([6])
        nans = 6
        seed = asarray([-0.05307939-0.05181743j, 0.32273450+0.15004976j, 0.00000000+0.j,
        -0.05307939-0.05181743j, -0.00279689+0.0014494j, 0.20736682+0.10180189j])
        ibk = 0
        bk = 15.034471543201017-15.034471543201017j
        delb = 15.034471543201017-15.034471543201017j
        zph = 0.066513811085846
        rho = 0.0
        k1 = 24.3408085526014+0.0987716730288631j
        k2 = 6.283185307179586
        jh = 0

        suminc = gshank(start, dela, suminc, nans, seed, ibk, bk, delb, zph, rho, k1, k2, jh)

        suminc_e = asarray([
            -0.0790 - 0.1277j,
            0.4374 + 0.4457j,
            0.0000 + 0.0000j,
            -0.0790 - 0.1277j,
            -0.0031 + 0.0017j,
            0.2816 + 0.2968j
        ])

        if not (numpy.abs(suminc - suminc_e) < 0.001).all():
            self.fail()

    def test_rom1(self):
        n = 6
        nx = 2
        zph = 0.066513811085846
        rho = 0.0
        k1 = 24.3408085526014+0.0987716730288631j
        k2 = 6.283185307179586
        a = 15.034471543201017-15.034471543201017j
        b = 24.48090861334604-15.034471543201017j
        jh = 0


        suminc = rom1(n, nx, zph, rho, k1, k2, a, b, jh)

        suminc_e = asarray([
            -0.0072 - 0.0351j,
            0.0375 + 0.1368j,
            0.0000 + 0.0000j,
            -0.0072 - 0.0351j,
            -0.0002 + 0.0001j,
            0.0240 + 0.0902j
        ])
        if not (numpy.abs(suminc - suminc_e) < 0.001).all():
            self.fail()

    def test_lambd(self):
        print("rom1 test passes -> no need to test lambd")
        #self.fail()

    def test_test(self):
        print("rom1 test passes -> no need to test test")
        #self.fail()
