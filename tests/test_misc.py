from unittest import TestCase
from numpy import *
from misc import *
from misc_io import *
from scatterer import *

__author__ = 'Kryosugarra'


class TestMisc(TestCase):
    def test_matchsize(self):
        a = asarray([1])
        b = asarray([2, 2, 2])
        a, b = matchsize(a, b)
        if not a.shape == b.shape:
            self.fail()

        a = asarray([1])
        b = asarray([2, 2, 2])
        c = asarray([3, 3])
        try:
            a, b, c = matchsize(a, b, c)
        except Exception as e:
            print(e)
            pass
        else:
            self.fail()

        a = asarray([1])
        b = asarray([2, 2, 2])
        c = asarray([3])
        a, b, c = matchsize(a, b, c)
        if (not a.shape == b.shape) or (not b.shape == c.shape) or (not a.shape == c.shape):
            self.fail()

    def test_threewide(self):
        onewide = asarray([1, 2, 3])
        twide = threewide(onewide)
        if not (twide == asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3]])).all():
            print(twide)
            self.fail()

        onewide = asarray([[1], [2], [3]])
        twide = threewide(onewide)
        if not (twide == asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3]])).all():
            print(twide)
            self.fail()

    def test_rtp2xyz(self):

        r = 1.0
        theta = 0
        phi = 0
        x, y, z = rtp2xyz(r, theta, phi)
        if not (x == 0. and y == 0. and z == 1.):
            self.fail()

        r = 1.0
        theta = pi / 4
        phi = pi / 4
        x, y, z = rtp2xyz(r, theta, phi)
        if not (x - cos(pi / 4) < 0.001 and y - cos(pi / 4) < 0.001 and z - cos(pi / 4) < 0.001):
            print(x, y, z)
            self.fail()

