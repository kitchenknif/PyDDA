from unittest import TestCase
from ott_funcs import *
from numpy import *

__author__ = 'Kryosugarra'


class TestMath(TestCase):
    def test_legendrerow(self):
        n = 0
        theta = pi
        x = legendrerow(n, theta)
        if not (numpy.abs(x - 0.2821) < 0.001):
            print(x)
            self.fail()

        n = 3
        theta = pi / 7
        x = legendrerow(n, theta)
        if not (numpy.abs(x - [0.3560, 0.4289, 0.1733, 0.0341]) < 0.01).all():
            print(x)
            self.fail()

            # n = 1
            # theta = [pi/7, pi/5, pi/2]
            # x = legendrerow(n, theta)
            # print(x)

    def test_sbesselh(self):
        n = 3
        kr = 5
        x = sbesselh(n, 1, kr)
        if not (numpy.abs(x - (0.2298 - 0.0154j)) < 0.001):
            self.fail()

        x = sbesselh(n, 2, kr)
        if not (numpy.abs(x - (0.2298 + 0.0154j)) < 0.001):
            self.fail()

        n = [1, 2, 3]
        x = sbesselh(n, 1, kr)
        if not (numpy.abs(x - [-0.0951 + 0.1804j, 0.1347 + 0.1650j, 0.2298 - 0.0154j]) < 0.001).all():
            print("x", x)
            self.fail()
        x = sbesselh(n, 2, kr)
        if not (numpy.abs(x - [-0.0951 - 0.1804j, 0.1347 - 0.1650j, 0.2298 + 0.0154j]) < 0.001).all():
            self.fail()

    def test_sbesselh1(self):
        n = 3
        kr = 5
        x = sbesselh1(n, kr)
        if not (numpy.abs(x - (0.2298 - 0.0154j)) < 0.001):
            self.fail()

        n = [1, 2, 3]
        x = sbesselh1(n, kr)
        if not (numpy.abs(x - [-0.0951 + 0.1804j, 0.1347 + 0.1650j, 0.2298 - 0.0154j]) < 0.001).all():
            self.fail()

    def test_sbesselh2(self):
        n = 3
        kr = 5
        x = sbesselh2(n, kr)
        if not (numpy.abs(x - (0.2298 + 0.0154j)) < 0.001):
            self.fail()

        n = [1, 2, 3]
        x = sbesselh2(n, kr)
        if not (numpy.abs(x - [-0.0951 - 0.1804j, 0.1347 - 0.1650j, 0.2298 + 0.0154j]) < 0.001).all():
            self.fail()

    def test_sbesselj(self):
        n = 8
        kr = 5.44
        x = sbesselj(n, kr)
        if not numpy.abs(x - 0.0099) < 0.001:
            print(x)
            self.fail()

    def test_spharm2(self):
        n = 3
        theta = pi / 7
        phi = pi / 8
        x = spharm2(n, theta, phi)
        if not (numpy.abs(
                    x - [-0.0130 + 0.0315j, 0.1226 - 0.1226j, -0.3963 + 0.1641j, 0.3560 + 0.0000j, 0.3963 + 0.1641j,
                         0.1226 + 0.1226j, 0.0130 + 0.0315j]) < 0.001).all():
            print(x)
            self.fail()

    def test_spharm(self):
        n = 3
        m = [1, 2, 3]
        theta = pi / 7
        phi = pi / 8
        x, y, y2 = spharm(n, m, theta, phi)
        if not (numpy.abs(x - [0.3963 + 0.1641j, 0.1226 + 0.1226j, 0.0130 + 0.0315j]) < 0.001).all():
            print("x", numpy.abs(x - [0.3963 + 0.1641, 0.1226 + 0.1226j, 0.0130 + 0.0315j]))
            self.fail()

        if not (numpy.abs(y - [0.3164 + 0.1311j, 0.4500 + 0.4500j, 0.0812 + 0.1961j]) < 0.001).all():
            print("y", numpy.abs(y - [0.3164 + 0.1311j, 0.4500 + 0.4500j, 0.0812 + 0.1961j]))
            self.fail()

        if not (numpy.abs(y2 - [-0.3783 + 0.9133j, -0.5650 + 0.5650j, -0.2177 + 0.0902j]) < 0.001).all():
            print("y2", numpy.abs(y2 - [-0.3783 + 0.9133j, -0.5650 + 0.5650j, -0.2177 + 0.0902j]))
            self.fail()

    def test_vsh(self):
        n = 3
        m = [1, 2, 3]
        theta = pi / 7
        phi = pi / 8
        x, y, y2 = vsh(n, m, theta, phi)
        if not (numpy.abs(x - [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.3164 + 0.1311j, 0.4500 + 0.4500j,
                               0.0812 + 0.1961j, -0.3783 + 0.9133j, -0.5650 + 0.5650j,
                               -0.2177 + 0.0902j]) < 0.001).all():
            print(x)
            self.fail()

        if not (numpy.abs(
                    y - [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, -0.3783 + 0.9133j, -0.5650 + 0.5650j,
                         -0.2177 + 0.0902j, -0.3164 - 0.1311j, -0.4500 - 0.4500j, -0.0812 - 0.1961j]) < 0.001).all():
            print(y)
            self.fail()

        if not (numpy.abs(
                    y2 - [0.3963 + 0.1641j, 0.1226 + 0.1226j, 0.0130 + 0.0315j, 0.0000 + 0.0000j, 0.0000 + 0.0000j,
                          0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j]) < 0.001).all():
            print(y2)
            self.fail()

    def test_vswf(self):
        n = 3
        m = 3
        kr = 3
        theta = pi / 7
        phi = pi / 8
        htype = 0
        M1, N1, M2, N2, M3, N3 = vswf(n, m, kr, theta, phi)

        if not numpy.abs(M1 - 0.0000 + 0.0000j) < 0.001:
            print("M1", numpy.abs(M1 - 0.0000 + 0.0000j))
            self.fail()
        if not numpy.abs(M2 - (0.0037 + 0.0359j)) < 0.001:
            print("M2", numpy.abs(M2 - 0.0037 + 0.0359j))
            self.fail()
        if not numpy.abs(M3 - (-0.0323 + 0.0033j)) < 0.001:
            print("M3", numpy.abs(M3 - -0.0323 + 0.0033j))
            self.fail()

        if not numpy.abs(N1 - (0.0208 - 0.0021j)) < 0.001:
            print("N1", numpy.abs(N1 - 0.0208 - 0.0021j))
            self.fail()
        if not numpy.abs(N2 - (-0.0102 + 0.0140j)) < 0.001:
            print("N2", numpy.abs(N2 - 0.0102 + 0.0140j))
            self.fail()
        if not numpy.abs(N3 - (-0.0155 - 0.0113j)) < 0.001:
            print("N3", numpy.abs(N3 - -0.0155 - 0.0113j))
            self.fail()

    def test_hankel0(self):
        z = 3 * numpy.pi / 7 + 9.88 * 1j
        x = hankel0(z)
        if not (numpy.abs(numpy.asarray(x) - [1.2622e-05 - 2.0139e-06j, 2.0300e-06 + 1.3248e-05j]) < 0.001).all():
            print(x)
            self.fail()

    def test_bessel0(self):
        z = 3 * numpy.pi / 7 + 9.88 * 1j
        x = bessel0(z)
        if not (numpy.abs(numpy.asarray(x) - [7.2467e+02 - 2.3933e+03j, -2.2764e+03 - 6.7054e+02j]) < 0.1).all():
            print(x)
            self.fail()
