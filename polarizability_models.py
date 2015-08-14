# Clausius-Mossoti & Lattice Dispersion Relations Polarizabilities
# Original author: Vincent Loke
# Affiliation: Physics Dept, School of Physical Sciences
#              The University of Queensland
# Date: 2007
#
# Ported to Python : Pavel Dmitriev
# Affiliation: Metamaterials Laboratory,
#              University of Information Technology, Mechanics and Optics,
#              St. Petersburg, Russia
# Date : 2015

import numpy
import numpy.linalg
import misc

def polarizability_CM(d, m):
    """Calculates Clausius-Mossoti polarizability of dipoles.

    Calcualtes Clausius-Mossoti Polarizability of dipole array according
    to their refractive indexes `m` and lattice spacing `d`.

    Parameters
    ----------
    d : float
        Dipole lattice spacing
    m : array_like
        List of dipole refractive indexes
    Returns
    -------
    alph: list
        List of dipole polarizabilities
    Notes
    -----
    Currently only supports isotropic polarizabilities,
    extending to anisotropic polarizabilities should be trivial,

    References
    ----------
    .. [1] Purcell, Edward M., and Carlton R. Pennypacker.
    "Scattering and absorption of light by nonspherical dielectric grains."
    The Astrophysical Journal 186 (1973): 705-714.

    """

    pow2 = misc.power_function(2)
    pow3 = misc.power_function(3)

    N = m.size
    msqr = pow2(m)
    dcube = pow3(d)

    alpha_CM = numpy.divide(numpy.multiply(msqr - 1, 3 * dcube / (4 * numpy.pi)), (msqr + 2))  # Clausius-Mossotti

    alph = numpy.zeros([3 * N], dtype=numpy.complex128)

    # assuming same polarizability in x, y & z directions
    for j in range(N):
        alph[3 * (j - 1) + 0] = alpha_CM[j]
        alph[3 * (j - 1) + 1] = alpha_CM[j]
        alph[3 * (j - 1) + 2] = alpha_CM[j]

    return alph

def polarizability_LDR(d, m, kvec, E0=None):
    """Calculates Lattice Dispersion Relation polarizability of dipoles.

    Calcualtes Lattice Dispersion Relation polarizability of dipole array according
    to their refractive indexes `m` and lattice spacing `d`.

    Parameters
    ----------
    d : float
        Dipole lattice spacing
    m : array_like
        List of dipole refractive indexes
    kvec : (3, 1) array_like
        Wave vector [kx ky kz]     e.g. [0 0 1] z-direction
    E0 : (3, 1) array_like
        E-field polarization [Ex Ey Ez]   e.g. [1 0 0] x-polarized,
        [1 i 0] left-handed circ pol.

    Returns
    -------
    alph: list
        List of dipole polarizabilities
    Notes
    -----
    Currently only supports isotropic polarizabilities,
    extending to anisotropic polarizabilities should be trivial,

    References
    ----------
    .. [1] Draine, Bruce T., and Jeremy Goodman. "Beyond Clausius-Mossotti-Wave propagation on a
    polarizable point lattice and the discrete dipole approximation."
    The Astrophysical Journal 405 (1993): 685-697.

    """

    pow2 = misc.power_function(2)
    pow3 = misc.power_function(3)

    k0 = 2 * numpy.pi
    N = m.size  # number of dipoles
    b1 = -1.8915316
    b2 = 0.1648469
    b3 = -1.7700004
    msqr = pow2(m)
    dcube = pow3(d)

    if E0 is not None:  # we have polarization info
        a_hat = kvec / numpy.linalg.norm(kvec)
        e_hat = E0 / numpy.linalg.norm(E0)
        S = 0
        for j in range(3):
            S += (a_hat[j] * e_hat[j]) ** 2
    else:  # use randomly-oriented value; also for non-plane wave
        S = .2

    alpha_CM = numpy.divide(3 * dcube / (4 * numpy.pi) * (msqr - 1), (msqr + 2))  # Clausius-Mossotti
    alpha_LDR = numpy.divide(alpha_CM, (1 + numpy.multiply((alpha_CM / dcube), (
        (b1 + msqr * b2 + msqr * b3 * S) * (k0 * d) ** 2 - 2 / 3 * 1j * k0 ** 3 * dcube))))

    alph = numpy.zeros([3 * N], dtype=numpy.complex128)
    # assuming same polarizability in x, y & z directions
    for j in range(N):
        alph[3 * (j - 1) + 0] = alpha_LDR[j]
        alph[3 * (j - 1) + 1] = alpha_LDR[j]
        alph[3 * (j - 1) + 2] = alpha_LDR[j]

    return alph
