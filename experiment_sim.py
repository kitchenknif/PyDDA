import disk_quadrature
import dda_si_funcs
import dda_funcs
import misc
import numpy

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

    weights = weights / (numpy.pi * pow2(maxradius))

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

    weights = weights / (numpy.pi * pow2(maxradius))

    I = 0
    for n in range(samples):
        for m in range(samples):
            I += pow2(k) * (rE[n*samples + m, 0]).T * numpy.dot(Esca[n*samples + m].conj(), Esca[n*samples + m])*weights[n]
    return I


