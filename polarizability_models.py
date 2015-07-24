# Clausius-Mossoti Polarizability

# Author: Vincent Loke
# Affiliation: Physics Dept, School of Physical Sciences
#              The University of Queensland
# Version: Pre-release (2007)
import numpy
import numpy.linalg


def polarizability_CM(d, m, k0):
    # m : N length vector containing relative refractive indices
    # % d : lattice spacing

    N = m.size  # number of dipoles
    msqr = m ** 2
    dcube = d ** 3

    alpha_CM = numpy.divide(numpy.multiply(msqr - 1, 3 * dcube / (4 * numpy.pi)), (msqr + 2))  # Clausius-Mossotti

    alph = numpy.zeros([3 * N], dtype=numpy.complex64)

    # assuming same polarizability in x, y & z directions
    for j in range(N):
        alph[3 * (j - 1) + 0] = alpha_CM[j]
        alph[3 * (j - 1) + 1] = alpha_CM[j]
        alph[3 * (j - 1) + 2] = alpha_CM[j]

    return alph


# Polarizability calculation based on Draine & Goodman,
# Beyond Clausius-Mossoti: wave propagation on a polarizable point lattice
# and the discrete dipole approximation,
# The Astrophysical Journal, 405:685-697, 1993 March 10

# Author: Vincent Loke
# Affiliation: Physics Dept, School of Physical Sciences
#              The University of Queensland
# Version: Pre-release (2007)
def polarizability_LDR(d, m, kvec, E0=None):
    # m : N length vector containing relative refractive indices
    #                                (isotropic version)
    # d : lattice spacing
    # kvec : wave vector [kx ky kz]     e.g. [0 0 1] z-direction

    # E0 : E-field polarization [Ex Ey Ez]   [1 0 0] x-polarized
    #                                        [1 i 0] left-handed circ pol.

    k0 = 2 * numpy.pi
    N = m.size  # number of dipoles
    b1 = -1.8915316
    b2 = 0.1648469
    b3 = -1.7700004
    msqr = m ** 2
    dcube = d ** 3

    if E0 is not None:  # we have polarization info
        a_hat = kvec / numpy.linalg.norm(kvec)
        e_hat = E0 / numpy.linalg.norm(E0)
        S = 0
        for j in range(3):
            S = S + (a_hat[j] * e_hat[j]) ** 2
    else:  # use randomly-oriented value; also for non-plane wave
        S = .2

    alpha_CM = numpy.divide(3 * dcube / (4 * numpy.pi) * (msqr - 1), (msqr + 2))  # Clausius-Mossotti
    alpha_LDR = numpy.divide(alpha_CM, (1 + numpy.multiply((alpha_CM / dcube), (
    (b1 + msqr * b2 + msqr * b3 * S) * (k0 * d) ** 2 - 2 / 3 * 1j * k0 ** 3 * dcube))))

    alph = numpy.zeros([3 * N], dtype=numpy.complex64)
    # assuming same polarizability in x, y & z directions
    for j in range(N):  # TODO TEST
        alph[3 * (j - 1) + 0] = alpha_LDR[j]
        alph[3 * (j - 1) + 1] = alpha_LDR[j]
        alph[3 * (j - 1) + 2] = alpha_LDR[j]

    return alph
