from unittest import TestCase
from disk_quadrature import *
from numpy import *
__author__ = 'Kryosugarra'


class TestDisk_disk_quadrature(TestCase):
    def test_disk_quadrature_rule(self):
        # for n in range(1, 11):
        #    w, r, t = disk_quadrature_rule(n, n)
        #    for i in range(n):
        #        print("{}, {}, {}, {}".format(i, w[i], r[i], t[i]))
        #    print("\n")

        n = 10
        w, r, t = disk_quadrature_rule(n, n)
        # for n in range(n):
        #     print("{}; w = {}; r = {}; t: {}".format(n, w[n], r[n], ", ".join([str(j) for j in t])))

        S = 0
        for i in range(n):
            for j in range(n):
                S += w[i]

        if not isclose(S, 1):
            self.fail()

    def test_legendre_ek_compute(self):
        n = 10
        x, w = legendre_ek_compute(n)
        # for i in range(n):
        #     print("{}, {}, {}".format(i, w[i], x[i]))

        w_exp = [0.0666713, 0.149451, 0.219086, 0.269267, 0.295524, 0.295524, 0.269267, 0.219086, 0.149451, 0.0666713]
        x_exp = [-0.973907, -0.865063, -0.67941, -0.433395, -0.148874, 0.148874, 0.433395, 0.67941, 0.865063, 0.973907]

        if not (numpy.allclose(w_exp, w) and numpy.allclose(x_exp, x)):
            self.fail()

    def test_imqlx(self):

        n = 5
        d = numpy.zeros(5)
        e = numpy.zeros(5)
        z = numpy.zeros(5)
        z[0] = sqrt(2.)
        for i in range(n):
            e[i] = ((i + 1) * (i + 1)) / (4 * (i + 1) * (i + 1) - 1)
            e[i] = numpy.sqrt(e[i])

        lam, qtz = imtqlx(d, e, z)

        lam_exp = [-0.90618, -0.538469, 4.77229e-017, 0.538469, 0.90618]
        qtz_exp = [-0.486751, -0.69183, -0.754247, 0.69183, 0.486751]

        # for i in range(n):
        #     print("{}, {}, {}".format(i, lam[i], qtz[i]))

        if not (allclose(lam, lam_exp) and allclose(qtz, qtz_exp)):
            self.fail()

        n = 5
        d = zeros(n)
        for i in range(0, n):
            d[i] = 2.0
        e = zeros(n)
        for i in range(0, n - 1):
            e[i] = -1.0
        e[n - 1] = 0.0
        z = ones(n)

        lam, qtz = imtqlx(d, e, z)

        lam2 = zeros(n)
        for i in range(0, n):
            angle = float(i + 1) * pi / float(2 * (n + 1))
            lam2[i] = 4.0 * (sin(angle)) ** 2

        if not allclose(lam, lam2):
            self.fail()
