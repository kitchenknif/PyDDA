from unittest import TestCase
from disk_quadrature import *
from numpy import *
__author__ = 'Kryosugarra'


class TestDisk_disk_quadrature(TestCase):

    def test_disk_quadrature_rule(self):
        for n in range(1, 11):
            w, r, t = disk_quadrature_rule(n, n)
            for i in range(n):
                print("{}, {}, {}, {}".format(i, w[i], r[i], t[i]))
            print("\n")
        #self.fail()


    def test_legendre_ek_compute(self):
        for n in range(1, 11):
            x, w = legendre_ek_compute(n)
            for i in range(n):
                print("{}, {}, {}".format(i, x[i], w[i]))
            print("\n")
        #self.fail()

    def test_imqlx(self):
        n = 5
        d = zeros (n)
        for i in range (n):
            d[i] = 2.0

        e = zeros (n)
        for i in range (n - 1):
            e[i] = -1.0
        e[n-1] = 0.0
        z = ones (n)

        lam, qtz = imtqlx (d, e, z )

        lam2 = zeros (n)
        for i in range (n):
            angle = float(i + 1) * pi / float(2 * (n + 1))
            lam2[i] = 4.0 * (sin(angle))**2

        for i in range(n):
            print("{}, {}, {}, {}, {}".format(i, lam[i], lam2[i], z[i], qtz[i]))


        #if not allclose(lam, lam2):
        #    self.fail()