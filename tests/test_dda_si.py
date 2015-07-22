from unittest import TestCase
from dda_si_funcs import *
from numpy import *
__author__ = 'Kryosugarra'


class TestDDA_SI(TestCase):
    def test_reflection_Rjk(self):
        self.fail()

    def test_interaction_AR(self):
        self.fail()

    def test_Fresnel_coeff_n(self):
        n = 1.3
        theta = pi/7

        r_te, r_tm = Fresnel_coeff_n(n, theta)
        if not ((r_te - (-0.1526)) < 0.001 and (r_tm - (-0.1081))):
            self.fail()

    def test_evanescent_k_e(self):
        self.fail()

    def test_evanescent_E(self):
        self.fail()

    def test_E_sca_SI(self):
        self.fail()