import numpy

from ott_funcs import *
from misc import *


def C_abs(k, E0, Ei, P, alph):
    # invalph = 1./alph;
    # C = 4*pi*k/abs(sum(E0.^2))*(-imag(dot(P,invalph.*P)) - 2/3*k^3*dot(P,P));

    C = 4 * numpy.pi * k / numpy.sum(numpy.abs(numpy.power(E0, 2))) * (-numpy.imag(numpy.vdot(P, numpy.divide(P, alph)))
                                                                       - (2. / 3.) * (k ** 3) * numpy.vdot(P, P))

    return C

    # just checking; got same result as above
    # C1 = 0;
    # N = length(P);
    # for j = 1:N
    #   C1 = C1 + (-imag(dot(P(j),P(j)/alph(j))) - 2/3*k^3*abs(P(j)^2));
    # end
    # %
    # % C1 = 4*pi*k/abs(sum(E0.^2))*C1


def C_ext(k, E0, Ei, P):
    # WARNING - this version is designed for Cartesian coordinates only

    C = numpy.divide(4 * numpy.pi * k, numpy.sum(numpy.abs(numpy.power(E0, 2)))) * numpy.imag(numpy.vdot(Ei, P))
    return C


# calculates a 3 X 3N block comprising N number of 3 X 3 Green's tensors
def calc_Aj(k, r, alph, j, blockdiag=True):
    pow2 = power_function(2)
    pow_m1 = power_function(-1)
    pow_m2 = power_function(-2)

    N, col = r.shape

    rk_to_rj = numpy.outer(numpy.ones([N]), r[j, :]) - r
    # rk_to_rj = numpy.tile(r[j, :], (N, 1)) - r
    # rk_to_rj = numpy.kron(numpy.ones([N, 1]), r[j, :]) - r

    Aj = numpy.zeros([3, 3 * N], dtype=numpy.complex128)  # vertical at first

    rjk = numpy.sqrt(numpy.sum(pow2(rk_to_rj), 1))

    rjk[j] = 1.  # Don't want divide-by-zero errors

    rjk_x = rk_to_rj[:, 0] / rjk
    rjk_x[j] = 0
    rjk_y = rk_to_rj[:, 1] / rjk
    rjk_y[j] = 0
    rjk_z = rk_to_rj[:, 2] / rjk
    rjk_z[j] = 0

    B = (1 - pow_m2(k * rjk) + 1j * pow_m1(k * rjk))
    B[j] = 0

    G = -(1 - 3 * pow_m2(k * rjk) + 1j * 3 * pow_m1(k * rjk))
    G[j] = 0

    C = -pow2(k) * numpy.exp(1j * k * rjk) / rjk
    C[j] = 0

    xy = C * G * rjk_x * rjk_y
    xz = C * G * rjk_x * rjk_z
    zy = C * G * rjk_y * rjk_z
    xx = C * (B + G * pow2(rjk_x))
    yy = C * (B + G * pow2(rjk_y))
    zz = C * (B + G * pow2(rjk_z))

    Aj[0, numpy.arange(0, 3 * N, 3)] = xx
    Aj[0, numpy.arange(1, 3 * N, 3)] = xy
    Aj[0, numpy.arange(2, 3 * N, 3)] = xz
    Aj[1, numpy.arange(0, 3 * N, 3)] = xy
    Aj[1, numpy.arange(1, 3 * N, 3)] = yy
    Aj[1, numpy.arange(2, 3 * N, 3)] = zy
    Aj[2, numpy.arange(0, 3 * N, 3)] = xz
    Aj[2, numpy.arange(1, 3 * N, 3)] = zy
    Aj[2, numpy.arange(2, 3 * N, 3)] = zz

    # block diagonal - inverse polarizability tensor
    if blockdiag:
        Aj[0, j * 3 + 0] = 1 / alph[j * 3 + 0]
        Aj[1, j * 3 + 1] = 1 / alph[j * 3 + 1]
        Aj[2, j * 3 + 2] = 1 / alph[j * 3 + 2]

    return Aj


def cross_C(C_vec):
    C = numpy.zeros([3, 3])
    C[0, 1] = -C_vec[2]
    C[0, 2] = C_vec[1]
    C[1, 0] = C_vec[2]
    C[1, 2] = -C_vec[0]
    C[2, 0] = -C_vec[1]
    C[2, 1] = C_vec[0]

    return C


def E_inc(E0, kvec, r):
    # E0: field amplitude [Ex Ey Ez]
    # kvec: wave vector, 2*pi in wavelength units
    # r: N x 3 matrix, for x_j, y_j, z_j coordinates
    # Following Smith & Stokes (2006), the result, E will have
    # N: number of dipoles
    # j = 1..N

    # E_inc_j = E_0 exp(ik.r_j - iwt)
    # Here, we omit the frequency factors exp(iwt) which can
    # be calculated outside this function if required. Thus
    # E_inc_j = E_0 exp(ik.r_j)

    N, cols = r.shape
    D = numpy.ones([N], dtype=numpy.complex128)
    kr = [numpy.dot([kvec[0] * D[i], kvec[1] * D[i], kvec[2] * D[i]], r[i, :]) for i in numpy.arange(D.size)]
    expikr = numpy.exp(numpy.multiply(1j, kr))
    # TODO: Get rid of asarray
    E1 = numpy.asarray([E0[0] * expikr, E0[1] * expikr, E0[2] * expikr], dtype=numpy.complex128).T  # N x 3

    # Ex, Ey & Ez components laid out into a 3N x 1 vector
    # Ei = [E1(:,1); E1(:,2); E1(:,3)];

    Ei = misc.col3to1(E1)

    return Ei


def E_inc_vswf(n, m, r_sp):
    global Ei_TE, Ei_TM, k

    N = r_sp.size
    M1, N1, M2, N2, M3, N3 = vswf(n, m, k * r_sp[:, 0], r_sp[:, 1], r_sp[:, 2])
    E_TE_sp = M3
    E_TM_sp = N3

    E_TE = numpy.zeros(3, N)
    E_TM = numpy.zeros(3, N)
    # convert to cartesian
    # for j = 1:N
    #   [E_TE(j,1),E_TE(j,2),E_TE(j,3)] = rtpv2xyzv(E_TE_sp(j,1),E_TE_sp(j,1),E_TE_sp(j,1),r_sp(j,1),r_sp(j,2),r_sp(j,3));
    #   [E_TM(j,1),E_TM(j,2),E_TM(j,3)] = rtpv2xyzv(E_TM_sp(j,1),E_TM_sp(j,1),E_TM_sp(j,1),r_sp(j,1),r_sp(j,2),r_sp(j,3));
    # end

    for j in range(1, N):
        theta = r_sp[j, 1]
        phi = r_sp[j, 2]
        M = [[numpy.sin(theta) * numpy.cos(phi), numpy.sin(theta) * numpy.sin(phi), numpy.cos(theta)],
             [numpy.cos(theta) * numpy.cos(phi), numpy.cos(theta) * numpy.sin(phi), -numpy.sin(theta)],
             [-numpy.sin(phi), numpy.cos(phi), 0]]

        E_TE[:, j] = numpy.transpose(M) * numpy.transpose(E_TE_sp[j, :])
        E_TM[:, j] = numpy.transpose(M) * numpy.transpose(E_TM_sp[j, :])

    Ei_TE = numpy.zeros(3 * N, 1)
    Ei_TM = numpy.zeros(3 * N, 1)

    for j in range(1, N):  # reformat into one column
        Ei_TE[3 * (j - 1) + 0] = E_TE[0, j]
        Ei_TE[3 * (j - 1) + 1] = E_TE[1, j]
        Ei_TE[3 * (j - 1) + 2] = E_TE[2, j]
        Ei_TM[3 * (j - 1) + 0] = E_TM[0, j]
        Ei_TM[3 * (j - 1) + 1] = E_TM[1, j]
        Ei_TM[3 * (j - 1) + 2] = E_TM[2, j]


def E_sca_FF(k, r, P, r_E):
    # k: wave number
    # r: dipole coordinates (N x 3 matrix)
    # P: polarizations (vector of length 3N; Px1,Py1,Pz1 ... PxN,PyN,PzN)
    # r_E: coord for the point at which to calculate the far field
    # Note: coordinates are relative to origin

    N, cols = r.shape

    E = 0
    r_norm = numpy.linalg.norm(r_E)
    r_hat = r_E / r_norm

    M = numpy.outer(r_hat.T, r_hat) - numpy.eye(3)
    for j in range(N):
        E = E + numpy.exp(-1j * k * numpy.vdot(r_hat, r[j, :])) * numpy.vdot(M, P[3 * j:3 * j + 3])

    E = E * (k ** 2) * numpy.exp(1j * k * r_norm) / r_norm

    return E


def interaction_A(k, r, alph, blockdiag=True):
    # global A

    N = r.shape[0]
    A = numpy.zeros([3 * N, 3 * N], dtype=numpy.complex128)
    # subj = 0;

    for j in range(N):
        # subj = subj + 1
        crow = 3 * j  # 3*(j-1)
        Aj = calc_Aj(k, r, alph, j, blockdiag)
        A[crow:crow + 3, :] = Aj

    return A
