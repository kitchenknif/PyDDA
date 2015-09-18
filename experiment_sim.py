import disk_quadrature
import dda_si_funcs
import dda_funcs
import polarizability_models
import scatterer
import misc
import numpy
import scipy.sparse.linalg


def objective_collection_si(k, dipoles, P, n1, NA, dist, samples=15):
    pow2 = misc.power_function(2)

    weights, r, theta = disk_quadrature.disk_quadrature_rule(samples, samples)

    alpha = numpy.arcsin(NA)
    maxradius = numpy.tan(alpha)*dist
    r *= maxradius

    rE = numpy.zeros([samples*samples, 3])
    for n in range(samples):
        for m in range(samples):
            rE[n*samples + m, 0] = numpy.sqrt(pow2(dist) + pow2(r[n]))
            rE[n*samples + m, 1] = theta[m]
            rE[n*samples + m, 2] = numpy.arctan(r[n]/dist)

    # calculate scattered field as a function of angles
    # parallel to incident plane

    #Esca = numpy.zeros([pow2(samples), 3])
    Esca = dda_si_funcs.E_sca_SI(k, dipoles, P, rE[:, 0], rE[:, 1], rE[:, 2], n1)

    weights /= numpy.pi * pow2(maxradius)

    I = 0
    for n in range(samples):
        for m in range(samples):
            I += pow2(k) * (rE[n*samples + m, 0]).T * numpy.dot(Esca[n*samples + m].conj(), Esca[n*samples + m])*weights[n]
    return I


def objective_collection(k, dipoles, P, NA, dist, samples=15):
    pow2 = misc.power_function(2)

    weights, r, theta = disk_quadrature.disk_quadrature_rule(samples, samples)

    alpha = numpy.arcsin(NA)
    maxradius = numpy.tan(alpha)*dist
    r *= maxradius

    rE = numpy.zeros([pow2(samples), 3])
    for n in range(samples):
        for m in range(samples):
            rE[n*samples + m, 0] = numpy.sqrt(pow2(dist) + pow2(r[n]))
            rE[n*samples + m, 1] = theta[m]
            rE[n*samples + m, 2] = numpy.arctan(r[n]/dist)

    # calculate scattered field as a function of angles

    Esca = numpy.zeros([pow2(samples), 3], dtype=numpy.complex128)
    for ix, r_e in enumerate(rE):
        r_E = numpy.zeros(3)
        r_E[0], r_E[1], r_E[2] = misc.rtp2xyz(r_e[0], r_e[1], r_e[2])
        Esca[ix, 0], Esca[ix, 1], Esca[ix, 2] = dda_funcs.E_sca_FF(k, dipoles, P, r_E)

    weights /= numpy.pi * pow2(maxradius)

    I = 0
    for n in range(samples):
        for m in range(samples):
            I += pow2(k) * (rE[n*samples + m, 0]).T * numpy.dot(Esca[n*samples + m].conj(), Esca[n*samples + m])*weights[n]
    return I


def scatter_intensity(r, n, d, E0, kvec, k, NA=0.45, samples=100):
        Ei = dda_funcs.E_inc(E0, kvec, r)  # direct incident field at dipoles
        alph = polarizability_models.polarizability_LDR(d, n, kvec)  # polarizability of dipoles
        A = dda_funcs.interaction_A(k, r, alph)
        P = scipy.sparse.linalg.gmres(A, Ei)[0]

        return objective_collection(k, r, P, NA, samples)


def scatter_spectrum(diameter, diameter_2, n_dipoles, refractive_index, lambda_range):
    pow1d3 = misc.power_function(1./3.)
    pow3 = misc.power_function(3)

    k = 2 * numpy.pi  # wave number
    # r, N, d_old = scatterer.dipole_sphere(10, diameter)
    r, N, d_old = scatterer.dipole_spheroid(n_dipoles, diameter, diameter_2, testsphere=False)

    I_scat_p = numpy.zeros(lambda_range.size)
    I_scat_s = numpy.zeros(lambda_range.size)

    # Incident plane wave
    # Incident field wave vector angle
    gamma_deg = 22.5
    gamma = gamma_deg / 180 * numpy.pi
    kvec = k * numpy.asarray([0, numpy.sin(gamma), -numpy.cos(gamma)])  # wave vector [x y z]

    n = refractive_index * numpy.ones([N])  # refractive index of sphere

    for i, lam in enumerate(lambda_range):
        d_new = pow1d3((4 / 3) * (numpy.pi / N) * (diameter/2 * diameter/2 * diameter_2/2 / pow3(lam)))
        r = (d_new/d_old) * r
        d_old = d_new

        #
        # p
        #
        E0 = numpy.asarray([1, 0, 0])  # E-field [x y z]  # s-pol
        I_scat_p[i] = scatter_intensity(r, n, d_new, E0, kvec, k)

        #
        # s
        #
        E0 = numpy.asarray([0, numpy.cos(gamma), numpy.sin(gamma)])  # p-pol
        I_scat_s[i] = scatter_intensity(r, n, d_new, E0, kvec, k)

    return I_scat_s, I_scat_p
