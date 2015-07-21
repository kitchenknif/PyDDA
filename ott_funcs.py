import numpy
import scipy.special
import misc

def legendrerow(n, theta):

    #x = scipy.special.legendre(n, theta)
    #return x.coeffs

    # legendrerow.m : Gives the spherical coordinate recursion in m for a given
    #                 n, theta.
    #
    # Usage:
    # pnm = legendrerow(n,theta)
    #
    # This provides approximately no benefit over the MATLAB implimentation. It
    # *may* provide a benefit in Octave. Inspiration from [Holmes and Featherstone, 2002]
    # and [Jekeli et al., 2007].
    #
    #% PACKAGE INFO

    if n==0:
        pnm = 1/numpy.sqrt(2*numpy.pi)/numpy.sqrt(2);
        return pnm

    theta = numpy.transpose(theta)

    ct=numpy.cos(theta)
    st=numpy.sin(theta)

    Wnn=numpy.sqrt((2*n+1)/(4*numpy.pi)*numpy.prod(1 -(1/2) * numpy.divide(1, range(1,n+1))) ) * numpy.ones(theta.shape) #first entry!

    #Wnn=numpy.sqrt((2*n+1)/(4*numpy.pi)*prod(1-1/2./[1:n]))*ones(size(theta)); #first entry!

    Wnnm1=numpy.sqrt(2*n) * ct * Wnn #second entry!
    lnm = len(range(0, n+1))

    pnm=numpy.zeros([lnm, theta.size])
    pnm[-1,:]=Wnn
    pnm[-2,:]=Wnnm1

    if lnm==2:
        pnm = [[Wnnm1], [Wnn]]
    else:
        jj=lnm-3
        for ii in range(n-2, -1, -1):
            a = numpy.sqrt(4*(ii+1)**2/(n-ii)/(n+ii+1))
            b = numpy.sqrt((n-ii-1)*(n+ii+2)/(n-ii)/(n+ii+1))

            pnm[jj, :] = a*ct*pnm[jj+1, :] - b * st**2 * pnm[jj+2, :]   #row recursion!
            jj = jj -  1

    ST,M = numpy.meshgrid(st,range(0,n+1))

    pnm = pnm * ST**M
    return pnm.T[0,:]  #TODO: Not sure if correct


def sbesselh(n, htype, kr):
    # sbesselh - spherical hankel function hn(kr)
    #
    # Usage:
    # hn = sbesselh(n,htype,kr)
    #
    # hn(kr) = sqrt(pi/2kr) Hn+0.5(kr)
    #
    # See besselh for more details
    #
    # PACKAGE INFO

    kr = numpy.transpose(kr)
    n = numpy.transpose(n)
    #hn = besselh(n'+1/2,htype,kr);
    if htype == 1:
        hn = scipy.special.hankel1(numpy.transpose(n)+1/2,kr)
    elif htype == 2:
        hn = scipy.special.hankel2(numpy.transpose(n)+1/2,kr)
    else:
        raise Exception("whoops")

    kr = numpy.kron(numpy.ones([n.size]), kr)
    hn = numpy.sqrt(numpy.kron(numpy.ones(kr.shape), numpy.pi) / (2*kr)) * hn

    return hn


def sbesselh1(n, kr):
    # sbesselh1 - spherical hankel function hn(kr) of the first kind
    #
    # Usage:
    # hn = sbesselh1(n,kr)
    #
    # hn(kr) = sqrt(pi/2kr) Hn+0.5(kr)
    #
    # See besselh for more details
    #
    # PACKAGE INFO

    kr = numpy.transpose(kr)
    n = numpy.transpose(n)
    hn = scipy.special.hankel1(numpy.transpose(n)+1/2,kr)
    kr = numpy.kron(numpy.ones([n.size]), kr)
    hn = numpy.sqrt(numpy.kron(numpy.ones(kr.shape), numpy.pi) / (2*kr)) * hn
    return hn


def sbesselh2(n, kr):
    # sbesselh1 - spherical hankel function hn(kr) of the second kind
    #
    # Usage:
    # hn = sbesselh1(n,kr)
    #
    # hn(kr) = sqrt(pi/2kr) Hn+0.5(kr)
    #
    # See besselh for more details
    #
    # PACKAGE INFO

    kr = numpy.transpose(kr)
    n = numpy.transpose(n)
    hn = scipy.special.hankel2(numpy.transpose(n)+1/2,kr)
    kr = numpy.kron(numpy.ones([n.size]), kr)
    hn = numpy.sqrt(numpy.kron(numpy.ones(kr.shape), numpy.pi) / (2*kr)) * hn
    return hn


def sbesselj(n, kr):
    #if isinstance(n, int) and isinstance(kr, int):
    #    return scipy.special.sph_jn(n, kr)[0][-1]
    #elif isinstance(n, int) and not isinstance(kr, int):
    #    return [scipy.special.sph_jn(n, kei)[0][-1] for kei in kr]
    #elif not isinstance(n, int):
    #    raise Exception('Not yet implemented')

    #return scipy.special.jn(n+0.5,kr)

    #function [jn] = sbesselj(n,kr)
    # sbesselj - spherical bessel function jn(kr)
    #
    # jn(kr) = sqrt(pi/2kr) Jn+0.5(kr)
    #
    # Usage: jn = sbessel(n,z);
    #
    # See besselj for more details
    #
    # PACKAGE INFO

    #kr=kr(:);
    #n=n(:);
    jn = scipy.special.jn(n + 1/2, kr)
    #n, kr = numpy.meshgrid(n, kr)

    kr = numpy.asarray(kr)
    n = numpy.asarray(n)

    small_args = kr[ numpy.abs(kr) < 1e-15]
    not_small_args =kr[ not (numpy.abs(kr) < 1e-15) ]

    if kr.size == 1 and numpy.abs(kr) < 1e-15:
         jn = numpy.divide(numpy.power(kr,n), numpy.prod(range(1,(2*n+2),2)))
    elif kr.size == 1 and not numpy.abs(kr) < 1e-15:
        jn = numpy.sqrt( numpy.pi / (2*kr)) * jn
    elif n.size == 1:
        jn[not_small_args] = numpy.sqrt(numpy.divide(numpy.pi, (2 * kr[not_small_args]))) * jn[not_small_args]
        jn[small_args] = numpy.divide(numpy.power(kr[small_args],n), numpy.prod(range(1(2*n+2),2)))
    else: # both n and kr are vectors
        jn[not_small_args] = numpy.sqrt(numpy.divide(numpy.pi, (2*kr[not_small_args]))) * jn[not_small_args]
        jn[small_args] = numpy.divide(numpy.power(kr[small_args],n[small_args]), [numpy.prod(range(1,(2*i+2),2)) for i in n[small_args]])

    return jn


def spharm(n, m, theta, phi):
    # spharm.m : scalar spherical harmonics and
    #            angular partial derivatives for given n,m (can take vector m).
    #
    # Usage:
    # Y = spharm(n,m,theta,phi)
    # or
    # [Y,dY/dtheta,1/sin(theta)*dY/dphi] = spharm(n,m,theta,phi)
    # or
    # Y = spharm(n,theta,phi)
    # or
    # [Y,dtY,dpY] = spharm(n,theta,phi)
    #
    # Scalar n for the moment.
    #
    # If scalar m is used Y is a vector of length(theta,phi) and is
    # completely compatible with previous versions of the toolbox. If vector m
    # is present the output will be a matrix with rows of length(theta,phi) for
    # m columns.
    #
    # "Out of range" n and m result in return of Y = 0
    #
    # PACKAGE INFO

    if not isinstance(n, int):
        raise Exception('n must be a scalar at present')

    #this is a cop out meant for future versions.
    #TODO Hopefully not used
    #if nargout > 1:
    #    mi = m
    #    m = range(-n,n)

    mi = m
    m = numpy.asarray(range(-n, n + 1)) #TODO TEST
    m = m[numpy.abs(m) <= n]

    theta, phi = misc.matchsize(numpy.asarray(theta), numpy.asarray(phi))
    input_length = theta.size

    # if abs(m) > n | n < 0
    #    Y = zeros(input_length,1);
    #    Ytheta = zeros(input_length,1);
    #    Yphi = zeros(input_length,1);
    #    return
    # end

    pnm = legendrerow(n, theta)
    #pnm = pnm(abs(m)+1,:).';

    # Why is this needed? Better do it, or m = 0 square integrals
    # are equal to 1/2, not 1.
    # This is either a bug in legendre or a mistake in the docs for it!
    # Check this if MATLAB version changes! (Version 5.X)
    # pnm(1,:) = pnm(1,:) * sqrt(2);

    pnm = pnm[abs(m)] #pick the m's we potentially have.

    phiM, mv = numpy.meshgrid(phi, m)

    pnm = numpy.append((numpy.power((-1), mv[m<0,:]).T * pnm[m < 0]), pnm[m >= 0])

    expphi = numpy.exp(1j*mv * phiM).T

    # N = sqrt((2*n+1)/(8*pi))

    Y = (pnm * expphi)[0]

    # We use recursion relations to find the derivatives, choosing
    # ones that don't involve division by sin or cos, so no need to
    # special cases to avoid division by zero

    # exp(i*phi),exp(-i*phi) are useful for both partial derivatives
    expplus = numpy.exp(numpy.multiply(1j, phiM))
    expminus = numpy.exp(numpy.multiply(-1j, phiM))

    # theta derivative
    # d/dtheta Y(n,m) = 1/2 exp(-i phi) sqrt((n-m)(n+m+1)) Y(n,m+1)
    #                 - 1/2 exp(i phi) sqrt((n-m+1)(n+m)) Y(n,m-1)

    ymplus = numpy.append(Y[1:], numpy.zeros([theta.size]) )

    ymminus = numpy.append( numpy.zeros([theta.size]), Y[0:-1] )

    Ytheta = numpy.sqrt((n-mv+1) * (n+mv)).T/2 * expplus.T * ymminus - numpy.sqrt((n-mv) * (n+mv+1)).T/2 * expminus.T * ymplus

    # phi derivative - actually 1/sin(theta) * d/dphi Y(n,m)
    # Note that this is just i*m/sin(theta) * Y(n,m), but we use a
    # recursion relation to avoid divide-by-zero trauma.
    # 1/sin(theta) d/dphi Y(n,m) =
    # i/2 * [ exp(-i phi) sqrt((2n+1)(n+m+1)(n+m+2)/(2n+3)) Y(n+1,m+1)
    #     + exp(i phi) sqrt((2n+1)(n-m+1)(n-m+2)/(2n+3)) Y(n+1,m-1) ]

    Y2 = (spharm2(n+1, theta, phi)).T

    ymplus=Y2[2:]
    ymminus=Y2[0:-2]

    #size(ymplus)
    #size(mv)
    #size(expminus)


    Yphi = 1j/2 * numpy.sqrt((2*n+1)/(2*n+3)) * ( numpy.sqrt((n+mv+1) * (n+mv+2)).T * expminus.T * ymplus + numpy.sqrt((n-mv+1) *(n-mv+2)).T * expplus.T * ymminus )

    Y=Y[numpy.add(mi, n)].T
    Yphi=Yphi[0, numpy.add(mi, n)].T
    Ytheta=Ytheta[0, numpy.add(mi, n)].T

    return Y,Ytheta,Yphi


def spharm2(n, theta, phi):
    # spharm.m : scalar spherical harmonics and
    #            angular partial derivatives for given n,m (can take vector m).
    #
    # Usage:
    # Y = spharm2(n,theta,phi)

    if not isinstance(n, int):
        raise Exception('n must be a scalar at present')

    m = numpy.asarray(range(-n, n+1)) #TODO TEST

    m = m[numpy.abs(m) <= n]

    theta, phi = misc.matchsize(numpy.asarray(theta), numpy.asarray(phi))
    input_length = theta.size

    # if abs(m) > n | n < 0
    #    Y = zeros(input_length,1);
    #    Ytheta = zeros(input_length,1);
    #    Yphi = zeros(input_length,1);
    #    return
    # end

    pnm = legendrerow(n, theta)
    #pnm = pnm(abs(m)+1,:).';

    # Why is this needed? Better do it, or m = 0 square integrals
    # are equal to 1/2, not 1.
    # This is either a bug in legendre or a mistake in the docs for it!
    # Check this if MATLAB version changes! (Version 5.X)
    # pnm(1,:) = pnm(1,:) * sqrt(2);

    pnm = pnm[abs(m)] #pick the m's we potentially have.

    phiM, mv = numpy.meshgrid(phi, m)

    pnm = numpy.append((numpy.power((-1), mv[m<0,:]).T * pnm[m < 0]), pnm[m >= 0])

    expphi = numpy.exp(1j*mv * phiM).T

    # N = sqrt((2*n+1)/(8*pi))

    Y = pnm * expphi

    #Y = numpy.transpose(Y)
    return Y[0]


def vsh(n, m, theta, phi):
    # vsh.m : Vector spherical harmonics. If m not specified will output for
    #           all m. Vector m input allowed.
    #
    # Usage:
    # [B,C,P] = vsh(n,m,theta,phi)
    # or
    # [B,C,P] = vsh(n,theta,phi)
    #
    # Scalar n for the moment.
    #
    # If scalar m: B,C,P are arrays of size length(theta,phi) x 3
    # If vector m: B,C,P are arrays of size length((theta,phi),m) x 3
    # theta and phi can be vectors (of equal length) or scalar.
    #
    # The three components of each vector are [r,theta,phi]
    #
    # "Out of range" n and m result in return of [0 0 0]
    #
    # PACKAGE INFO

    if not isinstance(n, int):
        raise Exception('n must be a scalar at present')

    #if phi is None:
    #    phi = theta
    #    theta = m
    #    m = range(-n, n+1) #TODO TEST

    # Convert a scalar theta or phi to a vector to match a vector
    # partner
    theta, phi = misc.matchsize(numpy.asarray(theta), numpy.asarray(phi))

    Y, Ytheta, Yphi = spharm(n, m, theta, phi)

    #this makes the vectors go down in m for n. has no effect if old version
    #code.

    Z = numpy.zeros(Y.size)

    B = numpy.append(numpy.append(Z, Ytheta), Yphi)

    C = numpy.append(numpy.append(Z, Yphi), -Ytheta)

    P = numpy.append(numpy.append(Y, Z), Z)

    return B, C, P


def vswf(n, m, kr, theta, phi, htype=0):

    # vswf.m : Vector spherical wavefunctions: M_k, N_k.
    #
    # Usage:
    # [M,N] = vswf(n,m,kr,theta,phi,type)
    # or
    # [M1,N1,M2,N2,M3,N3] = vswf(n,m,kr,theta,phi)
    # or
    # [M1,N1,M2,N2,M3,N3] = vswf(n,kr,theta,phi)
    #
    # where
    # kr, theta, phi are vectors of equal length, or scalar.
    # type = 1 -> outgoing solution - h(1)
    # type = 2 -> incoming solution - h(2)
    # type = 3 -> regular solution - j (ie RgM, RgN)
    #
    # Scalar n for the moment. If no type or m specified will calculate for all
    # types and m.
    #
    # M,N are arrays of size length(vector_input,m) x 3
    #
    # The three components of each vector are [r,theta,phi].
    #
    # "Out of range" n and m result in return of [0 0 0]
    #
    # PACKAGE INFO

    # Check input vectors
    # These must all be of equal length if non-scalar
    # and for good measure, we expand any scalar ones
    # to match the others in length

    if not isinstance(n, int):
        raise Exception('n must be a scalar at present')

    #if nargin < 5:
    #    htype=0;
    #    phi=theta;
    #    theta=kr;
    #    kr=m;
    #    m=[-n:n];

    #if nargin==5
    #    htype=0;


    #Convert all to column vectors
    #kr = kr.flatten(1)          # numpy.transpose(kr)
    #theta = theta.flatten(1)    # numpy.transpose(theta)
    #phi = phi.flatten(1)        # numpy.transpose(phi)

    #Check the lengths
    kr, theta, phi = misc.matchsize(numpy.asarray(kr), numpy.asarray(theta), numpy.asarray(phi))

    [B,C,P] = vsh(n, m, theta, phi)
    if n > 0:
        Nn = numpy.sqrt(1/(n*(n+1)))
    else:
        Nn = 0


    if htype == 1:
        if not isinstance(m, int):
            kr3 = numpy.kron(numpy.ones(1,m.size*3),kr) #makes all these suitable length
            hn  = numpy.kron(numpy.ones(1,m.size*3), sbesselh1(n,kr))
            hn1 = numpy.kron(numpy.ones(1,m.size*3),sbesselh1(n-1,kr))
        else:
            kr3=misc.threewide(kr) #makes all these suitable length
            hn=misc.threewide(sbesselh1(n,kr))
            hn1=misc.threewide(sbesselh1(n-1,kr))

        M = Nn * hn * C
        N = Nn * ( n*(n+1) / kr3 * hn * P + ( hn1 - n / kr3 * hn ) / B )
        M2 = 0
        N2 = 0
        M3 = 0
        N3 = 0

    elif htype == 2:

        if not isinstance(m, int):
            kr3  = numpy.kron(numpy.ones(1,m.size*3), kr)  #makes all these suitable length
            hn   = numpy.kron(numpy.ones(1,m.size*3),sbesselh2(n,kr))
            hn1  = numpy.kron(numpy.ones(1,m.size*3), sbesselh2(n-1,kr))
        else:
            kr3 = misc.threewide(kr); #makes all these suitable length
            hn  = misc.threewide(sbesselh2(n,kr))
            hn1 = misc.threewide(sbesselh2(n-1,kr))

        M = Nn * hn * C
        N = Nn * ( n*(n+1) / kr3 * hn * P + ( hn1 - n / kr3 * hn ) * B )
        M2 = 0
        N2 = 0
        M3 = 0
        N3 = 0
    elif htype == 3:

        if not isinstance(m, int):
            kr3 = numpy.kron(numpy.ones(1, m.size * 3), kr) #makes all these suitable length
            jn  = numpy.kron(numpy.kron(1,m.size * 3), sbesselj(n,kr))
            jn1 = numpy.kron(numpy.kron(1,m.size * 3), sbesselj(n-1,kr))
        else:
            kr3=misc.threewide(kr) #makes all these suitable length
            jn=misc.threewide(sbesselj(n,kr))
            jn1=misc.threewide(sbesselj(n-1,kr))


        M = Nn * jn * C
        N = Nn * ( n*(n+1) / kr3 * jn * P + ( jn1 - n / kr3 * jn  ) * B) #here is change!~!!!! get rid of jn->jn1
        M2 = 0
        N2 = 0
        M3 = 0
        N3 = 0

        if n != 1:
            N[kr3==0] = 0
        else:
            N[kr3==0] = 2/3 * Nn *( P[kr3==0] + B[kr3==0])

    else:
        if not isinstance(m, int):
            kr3 = numpy.kron(numpy.ones(1, m.size*3), kr)  #makes all these suitable length

            jn   = numpy.kron(numpy.ones(1,m.size*3), sbesselj(n,kr))
            jn1  = numpy.kron(numpy.ones(1,m.size*3),sbesselj(n-1,kr))

            hn1  = numpy.kron(numpy.ones(1,m.size*3),sbesselh1(n,kr))
            hn11 = numpy.kron(numpy.ones(1,m.size*3), sbesselh1(n-1,kr))

            hn2  = numpy.kron(numpy.ones(1,m.size*3), sbesselh2(n,kr))
            hn21 = numpy.kron(numpy.ones(1,m.size*3),sbesselh2(n-1,kr))
        else:
            kr3  = misc.threewide(kr) #makes all these suitable length

            hn2  = misc.threewide(sbesselh2(n, kr))
            hn21 = misc.threewide(sbesselh2(n-1, kr))

            hn1  = misc.threewide(sbesselh1(n, kr))
            hn11 = misc.threewide(sbesselh1(n-1, kr))

            jn   = misc.threewide(sbesselj(n, kr))
            jn1  = misc.threewide(sbesselj(n-1, kr))

        M = Nn * hn1 * C
        N = Nn * ( n*(n+1) / kr3 * hn1 * P + ( hn11 - n / kr3 * hn1 ) * B )
        M2 = Nn * hn2 * C
        N2 = Nn * ( n*(n+1) / kr3 * hn2 * P + ( hn21 - n / kr3 * hn2 ) * B )
        M3 = Nn * jn * C
        N3 = Nn * ( n*(n+1) / kr3 * jn * P + ( jn1 - n / kr3 * jn ) * B )

        if n != 1:
            N3[kr3==0] = 0
        else:
            N3[kr3==0] = 2/3 * Nn *( P[kr3==0] + B[kr3==0])

        M = M[0]
        N = N[0]

    return M[0], N[0], M[1], N[1], M[2], N[2]
    #return M, N, M2, N2, M3, N3

