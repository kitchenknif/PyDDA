#
# Based on LGPL code by John Burkardt.
# https://people.sc.fsu.edu/~jburkardt/
#
import numpy
import misc


def disk_quadrature_rule(nradial, nangles):
    """
    :param nradial: number of radial samples
    :param nangles: number of angle samples
    :return: w: radial weights
    :return: r: radial samples
    :return: t: angular samples
    """

    x, w = legendre_ek_compute(nradial)

    x = (x + 1.0)/2.0
    w /= 2.0

    r = numpy.zeros(nradial)
    t = numpy.zeros(nangles)

    for it in numpy.arange(nangles):
        t[it] = 2.0 * numpy.pi * it / nangles

    w /= nangles
    r = numpy.sqrt(x)

    return w, r, t

#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
def legendre_ek_compute(n):
    """
    Gauss-Legendre, Elhay-Kautsky method.
    :param n: number of samples
    :return: x, points
    :return: w, weights
    """

    pow2 = misc.power_function(2)

    zemu = 2.0

    bj = numpy.zeros(n)
    for i in numpy.arange(n):
        ip1 = i + 1.
        bj[i] = numpy.sqrt(pow2(ip1) / (4. * pow2(ip1) - 1.))

    x = numpy.zeros(n)
    w = numpy.zeros(n)
    w[0] = numpy.sqrt(zemu)

    x_r, w_r = imtqlx(x, bj, w)
    w_r = pow2(w_r)

    return x_r, w_r



#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#    Roger Martin, James Wilkinson,
#    The Implicit QL Algorithm,
#    Numerische Mathematik,
#    Volume 12, Number 5, December 1968, pages 377-383.
#
#    This routine is a slightly modified version of the EISPACK routine to
#    perform the implicit QL algorithm on a symmetric tridiagonal matrix.
#
#    The authors thank the authors of EISPACK for permission to use this
#    routine.
#
#    It has been modified to produce the product Q' * Z, where Z is an input
#    vector and Q is the orthogonal matrix diagonalizing the input matrix.
#    The changes consist (essentially) of applying the orthogonal
#    transformations directly to Z as they are generated.
# TODO: Fix fortran77-style indexes
def imtqlx(diagonal, subdiagonal, vec, abstol=1e-30, maxiterations = 30):
    """
    Diagonalize symmetric tridiagonal matrix
    :param diagonal: matrix diagonal, size N
    :param subdiagonal: matrix subdiagonal, size N-1
    :return: lam: diagonal entries of diagonalized matrix
    :return: qtz: values of Q' * Z, where Q diagonalizes input matrix M
    """

    n = diagonal.size
    #assert subdiagonal.size == n-1
    assert vec.size == n

    pow2 = misc.power_function(2)


    lam = numpy.zeros(n, dtype=numpy.complex128)
    lam = diagonal

    qtz = numpy.zeros(n, dtype=numpy.complex128)
    qtz = vec

    if n == 1:
        return lam, qtz

    subdiagonal[n-1] = 0

    for l in numpy.arange(1, n + 1):
        j = 0
        while True:
            for m in numpy.arange(l, n+1):
                if m == n:
                    break

                if numpy.abs(subdiagonal[m-1]) <= abstol * (numpy.abs(lam[m-1] + numpy.abs(lam[m]))):
                    break

            p = lam[l-1]

            if m == l:
                break

            if j >= maxiterations:
                raise Exception("Failed to converge in {} iterations".format(maxiterations))
            j += 1

            g = (lam[l] - p) / (2.0 * subdiagonal[l - 1])
            r = numpy.sqrt(pow2(g) + 1.0)

            if g < 0:
                t = g - r
            else:
                t = g + r
            g = lam[m-1] - p + subdiagonal[l-1] / (g + t)

            s = 1.0
            c = 1.0
            p = 0.0
            mml = m - l

            for ii in numpy.arange(1, mml + 1):
                i = m - ii
                f = s * subdiagonal[i - 1]
                b = c * subdiagonal[i - 1]

                if numpy.abs(g) <= numpy.abs(f):
                    c = g / f
                    r = numpy.sqrt(pow2(c) + 1.0)
                    subdiagonal[i] = f * r
                    s = 1.0 / r
                    c *= s
                else:
                    s = f / g
                    r = numpy.sqrt(pow2(s) + 1.0)
                    subdiagonal[i] = g * r
                    c = 1.0 / r
                    s *= c

                g = lam[i] - p
                r = (lam[i - 1] - g) * s + 2.0 * c * b
                p = s * r
                lam[i] = g + p
                g = c * r - b
                f = qtz[i]
                qtz[i] = s * qtz[i - 1] + c * f
                qtz[i - 1] = c * qtz[i - 1] - s * f

            lam[l - 1] -= p
            subdiagonal[l - 1] = g
            subdiagonal[m - 1] = 0.0

    #Sort
    for ii in numpy.arange(2, n + 1):
        i = ii - 1
        k = i
        p = lam[i - 1]

        for j in numpy.arange(ii, n + 1):

            if lam[j - 1] < p:
                k = j
                p = lam[j - 1]

        if k != i:
            lam[k - 1] = lam[i-1]
            lam[i-1] = p

            p = qtz[i - 1]
            qtz[i - 1] = qtz[k - 1]
            qtz[k - 1] = p

    return lam, qtz