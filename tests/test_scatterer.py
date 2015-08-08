from unittest import TestCase
import numpy

from scatterer import *
from misc import *

__author__ = 'Kryosugarra'


class TestScatterer(TestCase):
    def test_load_dda_si_dipole_file(self):
        pow2 = power_function(2)

        scat = Scatterer.load_dda_si_dipole_file('../shape/sphere_136.txt')
        r_max = 0
        for r in scat.dipoles:
            if numpy.sqrt(pow2(r[0]) + pow2(r[1]) + pow2(r[2])) > r_max:
                r_max = numpy.sqrt(pow2(r[0]) + pow2(r[1]) + pow2(r[2]))
        print(r_max)

        self.fail()

    def test_scatterer_from_shape(self):
        self.fail()

    def test_dipole_sphere(self):
        pow2 = power_function(2)

        scat = Scatterer.dipole_sphere(5, 1)
        r_max = 0
        for r in scat.dipoles:
            if numpy.sqrt(pow2(r[0]) + pow2(r[1]) + pow2(r[2])) > r_max:
                r_max = numpy.sqrt(pow2(r[0]) + pow2(r[1]) + pow2(r[2]))
        print(r_max)

        self.fail()

    def test_dipole_cube(self):
        scat = Scatterer.dipole_cube(4, 1)
        self.fail()
