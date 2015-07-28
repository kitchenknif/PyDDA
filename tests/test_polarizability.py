from unittest import TestCase
import numpy
from polarizability_models import *
__author__ = 'Kryosugarra'


class TestPolarizability_CM(TestCase):
    def test_polarizability_CM(self):
        alph_exp = 1.0e-04 *numpy.asarray([
           0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j,
            0.8351 + 0.2280j])

        k = 6.283185307179586
        m1 = 1.33 + .1j   # imag. component to demonstrate absorption
        k = 2*numpy.pi
        d = 1/(numpy.abs(m1)*k)
        m = m1*numpy.ones([8])

        alph=polarizability_CM(d, m, k)

        if not numpy.allclose(alph, alph_exp):
            self.fail()

    def test_polarizability_LDR(self):
        alph_exp = 1.0e-04 *numpy.asarray([
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j,
            0.8639 + 0.2599j])
        k = 6.283185307179586
        E0 = numpy.asarray([1, 1, 0])
        m1 = 1.33 + .1j;   # imag. component to demonstrate absorption
        k = 2*numpy.pi
        d = 1/(numpy.abs(m1)*k)
        m = m1*numpy.ones([8])
        kvec = [0, 0, k]

        alph=polarizability_LDR(d, m, kvec, E0)

        if not numpy.allclose(alph, alph_exp):
            self.fail()