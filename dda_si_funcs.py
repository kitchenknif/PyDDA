import numpy
import misc
from dda_si_integration_funcs import *


# % Reference:
# % Roland Schmehl, Brent M. Nebeker, and E. Dan Hirleman
# % "Discrete-dipole approximation for scattering by features on surfaces
# % by means of a two-dimensional fast Fourier transform technique"
# % J. Opt. Soc. Am. A 14, 3026-3036 (1997)
def reflection_Rjk(k1, k2, r_j, r_k, S):
    # % k1 = wave number in bottom medium (e.g. substrate)
    # % k2 = wave number in top medium (e.g. air/vacuum)
    # % r_j = receiving dipole coordinate [x y z]
    # % r_k = radiating dipole coordinate [x y z]
    # % S = precalculated Sommerfeld essential integrals

    pow2 = misc.power_function(2)
    pow_m1 = misc.power_function(-1)
    pow_m2 = misc.power_function(-2)

    k0 = 2 * numpy.pi

    # % (A9d)
    rI_k2j = [r_j[0] - r_k[0], r_j[1] - r_k[1], r_j[2] + r_k[2]]
    rIjk = numpy.linalg.norm(rI_k2j)

    # % x,y,z components vectors (A9a)
    rhat_x = rI_k2j[0] / rIjk
    rhat_y = rI_k2j[1] / rIjk
    rhat_z = rI_k2j[2] / rIjk

    # % scalars (A9b) & (A9c)
    beta = (1 - 1 * pow_m2(k0 * rIjk) + 1j * pow_m1(k0 * rIjk))
    gamma = -(1 - 3 * pow_m2(k0 * rIjk) + 1j * 3 * pow_m1(k0 * rIjk))

    zph = r_j[2] + r_k[2]
    rho = numpy.sqrt(pow2(r_j[0] - r_k[0]) + pow2(r_j[1] - r_k[1]))

    # % assign precalculated Sommerfeld essential integrals
    IV_rho = S[0]
    IV_z = S[1]
    IH_rho = S[2]
    IH_phi = S[3]
    #
    S11 = pow2(rhat_x) * IH_rho - pow2(rhat_y) * IH_phi
    S12 = rhat_x * rhat_y * (IH_rho + IH_phi)
    S13 = rhat_x * IV_rho
    S21 = rhat_x * rhat_y * (IH_rho + IH_phi)
    S22 = pow2(rhat_y) * IH_rho - pow2(rhat_x) * IH_phi
    S23 = rhat_y * IV_rho
    S31 = -rhat_x * IV_rho
    S32 = -rhat_y * IV_rho
    S33 = IV_z

    Sjk = numpy.asarray([[S11, S12, S13],
                         [S21, S22, S23],
                         [S31, S32, S33]])  # TODO: Get rid of asarray

    G11 = -(beta + gamma * pow2(rhat_x))
    G12 = -gamma * rhat_x * rhat_y
    G13 = gamma * rhat_x * rhat_z
    G21 = -gamma * rhat_y * rhat_x
    G22 = -(beta + gamma * pow2(rhat_y))
    G23 = gamma * rhat_y * rhat_z
    G31 = -gamma * rhat_z * rhat_x
    G32 = -gamma * rhat_z * rhat_y
    G33 = beta + gamma * pow2(rhat_z)

    Gjk = numpy.asarray([[G11, G12, G13],
                         [G21, G22, G23],
                         [G31, G32, G33]]) #TODO: Get rid of asarray
    # % to be consistent with the Draine and Flatau interraction matrix
    # % formulation we omit the (4*pi*ep)^-1 factor
    # % (A8)

    Rjk = -(Sjk + pow2(k0) * (pow2(k1) - pow2(k2)) / (pow2(k1) + pow2(k2)) * numpy.exp(
        1j * k0 * rIjk) / rIjk * Gjk)  # %as per Schmehl's thesis

    # % The required k0^2 factor was reported by Dr. Andrey Evlyukhin, Laser Zentrum Hannover e.V.
    # % as per Schmehl's thesis (2.41). In the previous version,
    # % Rjk = -(Sjk + (k1^2-k2^2)/(k1^2+k2^2)*exp(i*k0*rIjk)/rIjk*Gjk); % as per Schmehl (A8)
    return Rjk


def interaction_AR(k1, k2, r, alph):
    # % AR is 3N x 3N matrix
    # % k1 = wave number in bottom medium (substrate)
    # % k2 = wave number in top medium (could be air/vacuum)
    # % r: N x 3 matrix, for x_j, y_j, z_j coordinates
    # % N: number of dipoles
    # % j = 1..N
    # % AR means A + R
    # % A is as per Eqn(6) plus the inverse polarizability dagonal in
    # % Draine & Flatau,
    # % Discrete-dipole approximation for scattering calculations
    # % pgs 1491-1499, Vol. 11, No. 4 (1994), J. Opt. Soc. Am. A
    #
    # % the Rjk component represents radiating and receiving dipole interactions
    # % via substrate surface reflection; refer to:
    # % Roland Schmehl, Brent M. Nebeker, and E. Dan Hirleman
    # % "Discrete-dipole approximation for scattering by features on surfaces
    # % by means of a two-dimensional fast Fourier transform technique"
    # % J. Opt. Soc. Am. A 14, 3026-3036 (1997)

    pow2 = misc.power_function(2)
    pow_m1 = misc.power_function(-1)
    pow_m2 = misc.power_function(-2)

    k0 = 2 * numpy.pi
    N = r[:, 1].size
    AR = numpy.zeros([3 * N, 3 * N], dtype=numpy.complex128)
    I = numpy.eye(3)

    S, nS = precalc_Somm(r, k1, k2)

    iS = 0  # dipole combination counter
    # % nS(iS) determines which set of precalculated Sommmerfeld integrals to use
    # tic
    for jj in numpy.arange(N):
        # sprintf('Interaction matrix, dipole %d of %d',jj,N)
        for kk in numpy.arange(N):
            Rjk = reflection_Rjk(k1, k2, r[jj, :], r[kk, :], S[nS[iS], :])  # Schmehl et al.
            iS += 1
            if jj != kk:  # off-diagonal
                rk_to_rj = r[jj, :] - r[kk, :]
                rjk = numpy.linalg.norm(rk_to_rj)  # sqrt(sum((r(jj,:)-r(kk,:)).^2))
                rjk_hat = (rk_to_rj) / rjk
                rjkrjk = numpy.outer(rjk_hat, rjk_hat)

                Ajk = numpy.exp(1j * k0 * rjk) / rjk * (
                pow2(k0) * (rjkrjk - I) + (1j * k0 * rjk - 1) / pow2(rjk) * (3 * rjkrjk - I))  # %Draine & Flatau

                # % beta, gamma see Schmehl's thesis (1.27)
                rjk_x = rk_to_rj[0] / rjk
                rjk_y = rk_to_rj[1] / rjk
                rjk_z = rk_to_rj[2] / rjk
                beta = (1 - pow_m2(k0 * rjk) + 1j * pow_m1(k0 * rjk))
                gamma = -(1 - 3 * pow_m2(k0 * rjk) + 1j * 3 * pow_m1(k0 * rjk))
                Ajk_BG = -numpy.exp(1j * k0 * rjk) / rjk * pow2(k0) * numpy.asarray([
                    [(beta + gamma * pow2(rjk_x)), (gamma * rjk_x * rjk_y), (gamma * rjk_x * rjk_z)],
                    [(gamma * rjk_y * rjk_x), (beta + gamma * pow2(rjk_y)), (gamma * rjk_y * rjk_z)],
                    [(gamma * rjk_z * rjk_x), (gamma * rjk_z * rjk_y), (beta + gamma * pow2(rjk_z))]
                ])

                AR[jj * 3 + 0:(jj + 1) * 3, kk * 3 + 0:(kk + 1) * 3] = (Ajk + Rjk)
            else:
                AR[jj * 3 + 0, kk * 3 + 0] = 1. / alph[jj * 3 + 0]
                AR[jj * 3 + 1, kk * 3 + 1] = 1. / alph[jj * 3 + 1]
                AR[jj * 3 + 2, kk * 3 + 2] = 1. / alph[jj * 3 + 2]
                AR[jj * 3 + 0:(jj + 1) * 3, kk * 3 + 0:(kk + 1) * 3] += Rjk # + AR[(jj) * 3 + 0:(jj + 1) * 3, kk * 3 + 0:(kk + 1) * 3]
    return AR


# Fresnel amplitude reflection coefficients
def Fresnel_coeff_n(n_r, theta):
    # n_r : relative refractive index bottom substrate medium
    # theta : incident angle (from surface normal)

    pow2 = misc.power_function(2)

    R_TM = (numpy.sqrt(1 - pow2(numpy.sin(theta) / n_r)) - n_r * numpy.cos(theta)) / (
    numpy.sqrt(1 - pow2((numpy.sin(theta) / n_r))) + n_r * numpy.cos(theta))
    R_TE = (numpy.cos(theta) - n_r * numpy.sqrt(1 - pow2(numpy.sin(theta) / n_r))) / (
    numpy.cos(theta) + n_r * numpy.sqrt(1 - pow2(numpy.sin(theta) / n_r)))

    return R_TE, R_TM


# Calculates the wave vector of the evanescent field above a flat substrate
# whose interface on the x-y plane at z=0 (z being the vertical axis).
# The incident plane wave is internally reflected and its wave vectors are
# on the y-z plane, i.e., x=0

# Based on:
# Tojo, S. & Hasuo, M.,
# "Oscillator-strength enhancement of electric-dipole-forbidden transitions
# in evanescent light at total reflection",
# Physical Review A, 2005, 21, 012508

# However, the incident plane here is y-z instead of x-z.
def evanescent_k_e(theta_1, n1, n2):
    # theta_1 : incident angle
    # n1 : substrate reflective index
    # n2 : reflective index of the medium above the substrate, e.g., air

    # k2 : wave vector above the substrate
    # ep : TM polarisation vector
    # es : TE polarisation vector

    pow2 = misc.power_function(2)

    k0 = 2 * numpy.pi  # wave number in vacuum
    theta_c = numpy.arcsin(n2 / n1 + 0j)  # critical angle
    sin_theta_2 = (n1 / n2) * numpy.sin(theta_1)
    theta_2 = numpy.arcsin(sin_theta_2 + 0j)

    # This is correct regardless of incident angle. Has round-off error.
    # Use this for the general case
    # k2 = [0 n2*k0*sin_theta_2 n2*k0*cos(theta_2)];

    # This is only correct beyond the critical angle;
    # the sign for k_z is opposite otherwise. However, no round-off error.
    if numpy.isreal(theta_2):
        k2 = numpy.asarray([0, n1 * k0 * numpy.sin(theta_1), 1j * k0 * numpy.sqrt(pow2(n1 * numpy.sin(theta_1)) - pow2(n2))], dtype=numpy.complex128)
    else:
        k2 = numpy.asarray([0, n1 * k0 * numpy.sin(theta_1), -1j * k0 * numpy.sqrt(pow2(n1 * numpy.sin(theta_1)) - pow2(n2))], dtype=numpy.complex128)

    ep = numpy.asarray([0, 1j * numpy.sqrt(pow2(n1 / n2 * numpy.sin(theta_1)) - 1), n1 / n2 * numpy.sin(theta_1)], dtype=numpy.complex128)
    es = numpy.asarray([1, 0, 0], dtype=numpy.complex128)

    return k2, ep, es


# Calculates the evanescent field amplitudes above a flat substrate
# whose interface on the x-y plane at z=0 (z being the vertical axis).
# The incident plane wave is internally reflected and its wave vectors are
# on the y-z plane, i.e., x=0

# Based on:
# Tojo, S. & Hasuo, M.,
# "Oscillator-strength enhancement of electric-dipole-forbidden transitions
# in evanescent light at total reflection",
# Physical Review A, 2005, 21, 012508

# However, the incident plane here is y-z instead of x-z.
# The y-component of the incident and transmitted wave vector should be +ve
def evanescent_E(E1s, E1p, theta_1, n1, n2):
    # E1p : TM E-field amplitude
    # E1s : TE E-field amplitude
    # theta_1 : incident angle
    # n1 : substrate reflective index
    # n2 : reflective index of the medium above the substrate, e.g., air
    # k2 : wave vector above the substrate
    # E2p : TM E-field vector
    # E2s : TE E-field vector

    pow2 = misc.power_function(2)

    k2, ep, es = evanescent_k_e(theta_1, n1, n2)

    T2s = (2 * n1 * numpy.cos(theta_1)) / (
    n1 * numpy.cos(theta_1) + numpy.sqrt(pow2(n2) - pow2(n1 * numpy.sin(theta_1)) + 0j)) * E1s
    T2p = (2 * n1 * numpy.cos(theta_1)) / (
    n2 * numpy.cos(theta_1) + (n1 / n2) * numpy.sqrt(pow2(n2) - pow2(n1 * numpy.sin(theta_1)) + 0j)) * E1p

    E2s = es * T2s
    E2p = ep * T2p

    return k2, E2s, E2p


# % Draine & Flatau
#
def E_sca_SI(k, r, P, det_r, theta, phi, n1):
    # % k: wave vector
    # % r: dipole coordinates (N x 3 matrix)
    # % P: polarizations (vector of length 3N; Px1,Py1,Pz1 ... PxN,PyN,PzN)
    # % det_r, theta, phi: evaluation points in spherical coordinates
    # % n1: refractive index of substrate
    #
    # % Note: coordinates are relative to origin
    #
    pow2 = misc.power_function(2)

    N, cols = r.shape

    #TODO: what does this do?
    #rows, cols = theta.shape
    #if cols > rows:
    #    theta = numpy.reshape(theta, [cols, rows])
    #rows, cols = phi.shape
    #if cols > rows:
    #    phi = numpy.reshape(phi, [cols, rows])
    #rows, cols = det_r.shape
    #if cols > rows:
    #    det_r = numpy.reshape(det_r, [cols, rows])

    # %pts = length(r_unit);
    r_sp = numpy.asarray([det_r, theta, phi], dtype=numpy.complex128).T  # TODO: Get rid of asarray
    pts, cols = r_sp.shape
    r_unit = numpy.ones([pts], dtype=numpy.complex128)
    #er_sp = numpy.asarray([r_unit, theta, phi]).T
    #e1_sp = numpy.asarray([r_unit, theta + numpy.sign(theta) * numpy.pi / 2, phi]).T

    # TODO: Get rid of asarray
    r_E = numpy.asarray(misc.rtp2xyz(r_unit, theta, phi), dtype=numpy.complex128).T #er_sp
    r_E1 = numpy.asarray(misc.rtp2xyz(r_unit, theta + numpy.sign(theta) * numpy.pi / 2, phi), dtype=numpy.complex128).T #e1_sp
    er = numpy.zeros([pts, 3], dtype=numpy.complex128)
    e1 = numpy.zeros([pts, 3], dtype=numpy.complex128)
    e2 = numpy.zeros([pts, 3], dtype=numpy.complex128)

    # % r_E2 = rtp2xyz(er_sp);
    # % er2 = zeros(pts,3);
    for j in numpy.arange(pts):
        er[j, :] = r_E[j, :] / numpy.linalg.norm(r_E[j, :])
        e1[j, :] = r_E1[j, :] / numpy.linalg.norm(r_E1[j, :])
        e2[j, :] = numpy.cross(er[j, :], e1[j, :])

    E = numpy.zeros([pts, 3], dtype=numpy.complex128)

    for pt in numpy.arange(pts):
        erp = er[pt, :]  # e_r
        e1p = e1[pt, :]  # e_theta
        e2p = e2[pt, :]  # e_phi
        k_sca = (k * erp).conj()
        k_Isca = numpy.asarray([k_sca[0], k_sca[1], -k_sca[2]], dtype=numpy.complex128).conj()

        #   % This is the modification was made Mitchell Short, University of Utah,
        #   % to account for the Fresnel coefficient at each angle of dipole
        #   % as opposed to single incident angle.
        ref_angle = numpy.abs((-numpy.pi / 2) + pt * numpy.pi / theta.size)
        refl_TE, refl_TM = Fresnel_coeff_n(n1, numpy.abs(ref_angle))

        # %rp = norm(r_E(pt,:));
        rp = det_r[pt]
        for j in numpy.arange(N):
            Pj = P[3 * j + 0:3 * j + 3].conj()
            rj = r[j, :]
            rIj = [rj[0], rj[1], -rj[2]]

            E[pt, :] = E[pt, :] + numpy.exp(-1j * numpy.dot(k_sca, rj)) * (
            numpy.dot(Pj, e1p) * e1p + numpy.dot(Pj, e2p) * e2p) + numpy.exp(-1j * numpy.dot(k_Isca, rj)) * (
            refl_TM * numpy.dot(Pj, e1p) * e1p + refl_TE * numpy.dot(Pj, e2p) * e2p)
            # TODO: Inner Outer product?

            # E(pt,:) = E(pt,:) + ...
            # exp(-i*k_sca.*rj).*(dot(Pj,e1p)*e1p + dot(Pj,e2p)*e2p) + ...
            # exp(-i*k_sca.*rIj).*(refl_TM*dot(Pj,e1p)*e1p + refl_TE*dot(Pj,e2p)*e2p);

            # % dot(k_sca,rIj) and dot(k_Isca,rj) are equal.
            # % Refer to Schmel's thesis (2.58)
        E[pt, :] = E[pt, :] * pow2(k) * numpy.exp(1j * k * rp) / (4 * numpy.pi * rp)
    return E
