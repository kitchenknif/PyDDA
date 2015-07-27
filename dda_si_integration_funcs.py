import numpy
import misc
from ott_funcs import *


# # gshank integrates the 6 sommerfeld integrals from start to
# # infinity (until convergence) in lambda.  at the break point, bk,
# # the step increment may be changed from dela to delb.  shank's
# # algorithm to accelerate convergence of a slowly converging series
# # is used
# # void gshank( complex double start, complex double dela, complex double *sum,
# #     int nans, complex double *seed, int ibk, complex double bk, complex double delb )
def gshank(start, dela, suminc, nans, seed, ibk, bk, delb, zph, rho, k1, k2, jh):  # return suminc
    # # start = A_0
    # # dela = delta before break point
    # # suminc = increment of integral
    # # nans = number of functions
    # # seed = S_0
    # # ibk = flag if path contains a break point (1 or 0)
    # # bk = break point (d*) in the path
    # # delb = delta after break point
    # # jh = flag 1 or 0, respectively for Bessel or Hankel function forms
    #
    # # complex double q1[6][20], q2[6][20], ans1[6], ans2[6];

    #suminc = suminc_start

    q1 = numpy.zeros([6, 20], dtype=numpy.complex128)
    q2 = numpy.zeros([6, 20], dtype=numpy.complex128)
    ans1 = numpy.zeros([6, 1], dtype=numpy.complex128)

    # # constants
    MAXH = 20
    CRIT = 1E-4

    rbk = numpy.real(bk)
    delt = dela
    if ibk == 0:
        ibx = 1
    else:
        ibx = 0

    #brk = False  # TODO: Where should it be assigned?
    # for( i = 0; i < nans; i++ )
    # #   ans2[i]=seed[i];
    ans2 = seed

    b = start
    # for( intx = 1; intx <= MAXH; intx++ )
    for intx in numpy.arange(1, MAXH + 1):  # TODO: test interval
        inx = intx - 1  # inx=intx-1
        a = b
        b = b + delt  # b += del

        if (ibx == 0) and (numpy.real(b) >= rbk):
            # hit break point.  reset seed and start over.
            ibx = 1
            b = bk
            delt = delb
            suminc = rom1(nans, 2, zph, rho, k1, k2, a, b, jh)  # rom1(nans,sum,2)
            if ibx != 2:
                # for( i = 0; i < nans; i++ )
                #     ans2[i] += sum[i];
                ans2 = ans2 + suminc
                intx = 0
                continue

            # #   for( i = 0; i < nans; i++ )
            # #     ans2[i]=ans1[i]+sum[i];
            ans2 = ans1 + suminc
            intx = 0
            continue
            # end #  } /* if( (ibx == 0) && (creal(b) >= rbk) ) */

        suminc = rom1(nans, 2, zph, rho, k1, k2, a, b, jh)  # rom1(nans,sum,2);
        # # for( i = 0; i < nans; i++ )
        # #   ans1[i] = ans2[i]+sum[i];
        ans1 = ans2 + suminc

        a = b
        b = b + delt  # b += del;

        # # if( (ibx == 0) && (creal(b) >= rbk) )
        if (ibx == 0) and (numpy.real(b) >= rbk):
            # hit break point.  reset seed and start over.
            ibx = 2
            b = bk
            delt = delb
            suminc = rom1(nans, 2, zph, rho, k1, k2, a, b, jh)  # rom1(nans,sum,2);
            if ibx != 2:
                # for( i = 0; i < nans; i++ )
                #     ans2[i] += sum[i];
                ans2 = ans2 + suminc
                intx = 0
                continue

            # for( i = 0; i < nans; i++ )
            #     ans2[i] = ans1[i]+sum[i];
            ans2 = ans1 + suminc
            intx = 0
            continue
            # end # } /* if( (ibx == 0) && (creal(b) >= rbk) ) */

        suminc = rom1(nans, 2, zph, rho, k1, k2, a, b, jh)  # rom1(nans,sum,2);

        # for( i = 0; i < nans; i++ )
        #   ans2[i]=ans1[i]+sum[i];
        ans2 = ans1 + suminc

        den = 0
        for j in numpy.arange(nans):  # for( i = 0; i < nans; i++ )
            as1 = ans1[j]  # as1=ans1[i];
            as2 = ans2[j]  # as2=ans2[i];

            if intx >= 2:
                # for( j = 1; j < intx; j++ )
                for k in numpy.arange(1, intx):
                    km = k - 1  # jm=j-1;
                    aa = q2[j, km]  # aa=q2[i][jm];
                    a1 = q1[j, km] + as1 - 2. * aa  # a1=q1[i][jm]+as1-2.*aa;

                    # if( (creal(a1) != 0.) or (cimag(a1) != 0.) ):
                    if (numpy.real(a1) != 0.) or (numpy.imag(a1) != 0.):
                        a2 = aa - q1[j, km]  # a2=aa-q1[i][jm];
                        a1 = q1[j, km] - a2 * a2 / a1  # a1=q1[i][jm]-a2*a2/a1;
                    else:
                        a1 = q1[j, km]  # a1=q1[i][jm];

                    a2 = aa + as2 - 2 * as1
                    # if( (creal(a2) != 0.) || (cimag(a2) != 0.) )
                    if (numpy.real(a2) != 0) or (numpy.imag(a2) != 0):
                        a2 = aa - (as1 - aa) * (as1 - aa) / a2
                    else:
                        a2 = aa

                    q1[j, km] = as1  # q1[i][jm]=as1;
                    q2[j, km] = as2  # q2[i][jm]=as2;
                    as1 = a1
                    as2 = a2
                    # end # } /* for( j = 1; i < intx; i++ ) */
                    # end # } /* if(intx >= 2) */

            q1[j, intx - 1] = as1  # q1[i][intx-1]=as1;
            q2[j, intx - 1] = as2  # q2[i][intx-1]=as2;
            # amg=fabs(creal(as2))+fabs(cimag(as2));
            amg = numpy.abs(numpy.real(as2)) + numpy.abs(numpy.imag(as2))
            if amg > den:
                den = amg

            # end # } /* for( i = 0; i < nans; i++ ) */

        denm = 1E-3 * den * CRIT  # denm=1.e-3*den*CRIT;
        km = intx - 3  # jm=intx-3;
        if km < 1:
            km = 1

        # #for( j = jm-1; j < intx; j++ )
        for k in numpy.arange(km - 1, intx):
            brk = False  # brk = FALSE;
            # for (i = 0; i < nans; i++ )
            for j in numpy.arange(nans):
                a1 = q2[j, k]  # a1=q2[i][j];
                den = (numpy.abs(numpy.real(a1)) + numpy.abs(
                    numpy.imag(a1))) * CRIT  # den=(fabs(creal(a1))+fabs(cimag(a1)))*CRIT;
                if den < denm:
                    den = denm

                a1 = q1[j, k] - a1  # a1=q1[i][j]-a1;
                amg = numpy.abs(numpy.real(a1) + numpy.abs(numpy.imag(a1)))
                if amg > den:
                    brk = True  # brk = TRUE;
                    break

            # end # } /* for( i = 0; i < nans; i++ ) */

            if brk:
                break
                # end # } /* for( j = jm-1; j < intx; j++ ) */

        if not brk:  # if( ! brk )
            # for( i = 0; i < nans; i++ )
            #     sum[i]=.5*(q1[i][inx]+q2[i][inx]);
            suminc = .5 * (q1[:, inx] + q2[:, inx])
            return suminc

        # end # } /* for( intx = 1; intx <= maxh; intx++ ) */

    raise Exception("Did not converge! :(")
    # #   /* No convergence */
    # #   abort_on_error(-6);
    # # TODO: handle error


# % rom1 integrates the 6 sommerfeld integrals from a to b in lambda.
# % the method of variable interval width romberg integration is used.
# % void rom1( int n, complex double *sum, int nx )
def rom1(n, nx, zph, rho, k1, k2, a, b, jh):
    # % n = number of functions?
    # % nx = ?
    # % static double z, ze, s, ep, zend, dz=0., dzot=0., tr, ti;
    # % static complex double t00, t11, t02;
    # % static complex double g1[6], g2[6], g3[6], g4[6], g5[6], t01[6], t10[6], t20[6];

    dz = 0
    dzot = 0
    t01 = numpy.zeros([n], dtype=numpy.complex128)
    t10 = numpy.zeros([n], dtype=numpy.complex128)
    t20 = numpy.zeros([n], dtype=numpy.complex128)

    # % constants
    NM = 131072
    CRIT = 1E-4
    NTS = 4

    lstep = 0
    z = 0
    ze = 1
    s = 1
    ep = s / (1E4 * NM)  # ep=s/(1.e4*NM)
    zend = ze - ep
    # %   for( i = 0; i < n; i++ )
    # %     sum[i]=CPLX_00;

    suminc = numpy.zeros([n])
    ns = nx
    nt = 0
    g1 = saoa(z, zph, rho, k1, k2, a, b, jh)  # saoa(z,g1)

    jump = 0  # jump = FALSE;
    while True:  # while( TRUE )
        if not jump:  # if( ! jump )
            dz = s / ns
            if (z + dz) > ze:
                dz = ze - z
                if dz <= ep:
                    return suminc
            dzot = dz * .5
            g3 = saoa(z + dzot, zph, rho, k1, k2, a, b, jh)  # saoa(z+dzot,g3)
            g5 = saoa(z + dz, zph, rho, k1, k2, a, b, jh)  # saoa(z+dz,g5)

        # /* if( ! jump ) */

        nogo = 0  # nogo = FALSE;

        # %   for( i = 0; i < n; i++ )
        # %   {
        # %     t00=(g1[i]+g5[i])*dzot;
        # %     t01[i]=(t00+dz*g3[i])*.5;
        # %     t10[i]=(4.*t01[i]-t00)/3.;
        # %
        # %     /* test convergence of 3 point romberg result */
        # %     test( creal(t01[i]), creal(t10[i]), &tr, cimag(t01[i]), cimag(t10[i]), &ti, 0. );
        # %     if( (tr > CRIT) || (ti > CRIT) )
        # % 	    nogo = TRUE;
        # %   }

        for j in numpy.arange(n):
            t00 = (g1[j] + g5[j]) * dzot
            t01[j] = (t00 + dz * g3[j]) * .5
            t10[j] = (4 * t01[j] - t00) / 3

            # test convergence of 3 point romberg result
            # test( creal(t01[i]), creal(t10[i]), &tr, cimag(t01[i]), cimag(t10[i]), &ti, 0. );
            tr, ti = test(numpy.real(t01[j]), numpy.real(t10[j]), numpy.imag(t01[j]), numpy.imag(t10[j]), 0)
            if (tr > CRIT) or (ti > CRIT):
                nogo = 0  # nogo=TRUE TODO: TEST WHY NOT 1

        if not nogo:  # if( ! nogo ):
            # for( i = 0; i < n; i++ )
            #     sum[i] += t10[i];
            suminc = suminc + t10

            nt = nt + 2  # nt += 2
            z = z + dz  # z += dz
            if z > zend:
                return suminc
            # for( i = 0; i < n; i++ )
            #     g1[i]=g5[i];
            g1 = g5

            if (nt >= NTS) and (ns > nx):
                ns = ns / 2
                nt = 1

            jump = 0  # jump = FALSE
            continue
            # }  /* if( ! nogo ) */

        g2 = saoa(z + dz * .25, zph, rho, k1, k2, a, b, jh)  # saoa(z+dz*.25,g2)
        g4 = saoa(z + dz * .75, zph, rho, k1, k2, a, b, jh)  # saoa(z+dz*.75,g4)
        nogo = 0  # nogo=FALSE;

        # % for( i = 0; i < n; i++ )
        # % {
        # %   t02=(t01[i]+dzot*(g2[i]+g4[i]))*.5;
        # %   t11=(4.*t02-t01[i])/3.;
        # %   t20[i]=(16.*t11-t10[i])/15.;
        # %
        # %   /* test convergence of 5 point romberg result */
        # %   test( creal(t11), creal(t20[i]), &tr, cimag(t11), cimag(t20[i]), &ti, 0. );
        # %   if( (tr > CRIT) || (ti > CRIT) )
        # %   nogo = TRUE;
        # % }

        for j in numpy.arange(n):
            t02 = (t01[j] + dzot * (g2[j] + g4[j])) * .5
            t11 = (4 * t02 - t01[j]) / 3
            t20[j] = (16 * t11 - t10[j]) / 15  # TODO: TEST WHY I NOT J

            # test convergence of 5 point romberg result
            tr, ti = test(numpy.real(t11), numpy.real(t20[j]), numpy.imag(t11), numpy.imag(t20[j]),
                          0)  # TODO: TEST WHY I NOT J
            if (tr > CRIT) or (ti > CRIT):
                nogo = 1  # nogo = TRUE;

        if not nogo:  # if( ! nogo )
            # for( i = 0; i < n; i++ )
            #     sum[i] += t20[i];
            suminc = suminc + t20

            nt += 1  # nt++
            z = z + dz  # z += dz
            if z > zend:
                return suminc

            # for( i = 0; i < n; i++ )
            #     g1[i]=g5[i];

            g1 = g5

            if (nt >= NTS) and (ns > nx):
                ns = ns / 2
                nt = 1

            jump = 0  # jump = FALSE;
            continue
            # end % } /* if( ! nogo ) */

        nt = 0
        if ns < NM:
            ns = ns * 2  # ns *= 2;
            dz = s / ns
            dzot = dz * .5

            # %     for( i = 0; i < n; i++ )
            # %     {
            # % 	    g5[i]=g3[i];
            # % 	    g3[i]=g2[i];
            # %     }
            g5 = g3
            g3 = g2

            jump = 1  # jump = TRUE;
            continue
            # end % } /* if(ns < nm) */

        if not lstep:  # if( ! lstep )
            lstep = 1  # lstep = TRUE;
            t00, t11 = lambd(z, a, b)  # lambda( z, &t00, &t11 );

        # %   for( i = 0; i < n; i++ )
        # %     sum[i] += t20[i];
        suminc = suminc + t20
        nt += 1  # nt++
        z += dz  # z += dz

        if z > zend:
            return suminc

        # %   for( i = 0; i < n; i++ )
        # %     g1[i]=g5[i];
        g1 = g5

        if (nt >= NTS) and (ns > nx):
            ns = ns / 2  # ns /= 2;
            nt = 1

        jump = 0  # jump = FALSE;
        # end %%  } /* while( TRUE ) */


# % saoa computes the integrand for each of the 6 sommerfeld */
# % integrals for source and observer above ground */
# %void saoa( double t, complex double *ans)
def saoa(t, zph, rho, k1, k2, a, b, jh):
    # %   double xlr, sign;
    # %   static complex double xl, dxl, cgam1, cgam2, b0, b0p, com, dgam, den1, den2;

    pow2 = misc.power_function(2)
    pow4 = misc.power_function(4)
    pow6 = misc.power_function(6)

    tsmag = 100 * k1 * numpy.conj(k1)
    cksm = pow2(k2) / (pow2(k1) + pow2(k2))  # cksm=ck2sq/(ck1sq+ck2sq);
    ct1 = .5 * (pow2(k1) - pow2(k2))  # ct1=.5*(ck1sq-ck2sq)
    # % erv=ck1sq*ck1sq;
    # % ezv=ck2sq*ck2sq;
    # % ct2=.125*(erv-ezv);
    ct2 = .125 * (pow4(k1) - pow4(k2))
    # % erv *= ck1sq;
    # % ezv *= ck2sq;
    # % ct3=.0625*(erv-ezv);
    ct3 = .0625 * (pow6(k1) - pow6(k2))

    xl, dxl = lambd(t, a, b)  # lambda(t, &xl, &dxl);

    if jh == 0:
        # bessel function form
        b0, b0p = bessel0(xl * rho)  # bessel(xl*rho, &b0, &b0p);
        b0 *= 2.  # b0  *=2.
        b0p *= 2.  # b0p *=2.
        cgam1 = numpy.sqrt(pow2(xl) - pow2(k1))  # cgam1=csqrt(xl*xl-ck1sq)
        cgam2 = numpy.sqrt(pow2(xl) - pow2(k2))  # cgam2=csqrt(xl*xl-ck2sq)
        if numpy.real(cgam1) == 0:  # if(creal(cgam1) == 0.):
            cgam1 = numpy.abs(numpy.imag(cgam1)) * 1j  # cgam1=cmplx(0.,-fabs(cimag(cgam1)))
        if numpy.real(cgam2) == 0:  # if(creal(cgam2) == 0.):
            cgam2 = numpy.abs(numpy.imag(cgam2)) * 1j  # cgam2=cmplx(0.,-fabs(cimag(cgam2)))
    else:
        # hankel function form
        b0, b0p = hankel0(xl * rho)  # hankel(xl*rho, &b0, &b0p);
        com = xl - k1  # com=xl-ck1;
        cgam1 = numpy.sqrt(xl + k1) * numpy.sqrt(com)  # cgam1=csqrt(xl+ck1)*csqrt(com);
        if (numpy.real(com) < 0) and (numpy.imag(com) >= 0):  # if(creal(com) < 0. && cimag(com) >= 0.)
            cgam1 = -cgam1

        com = xl - k2  # com=xl-ck2;
        cgam2 = numpy.sqrt(xl + k2) * numpy.sqrt(com)  # cgam2=csqrt(xl+ck2)*csqrt(com);
        if (numpy.real(com) < 0) and (numpy.imag(com) >= 0):  # if(creal(com) < 0. && cimag(com) >= 0.)
            cgam2 = -cgam2

    xlr = xl * numpy.conj(xl)
    if xlr >= tsmag:
        if numpy.imag(xl) >= 0:
            xlr = numpy.real(xl)
            if xlr >= k2:
                if (xlr <= numpy.realk1):  # if(xlr <= ck1r):
                    dgam = cgam2 - cgam1
                else:
                    sign = 1
                    dgam = 1 / (pow2(xl))  # dgam=1./(xl*xl)
                    dgam = sign * ((ct3 * dgam + ct2) * dgam + ct1) / xl
            else:
                sign = -1
                dgam = 1 / pow2(xl)  # dgam=1./(xl*xl)
                dgam = sign * ((ct3 * dgam + ct2) * dgam + ct1) / xl
                # /* if(xlr >= ck2) */
                # /* if(cimag(xl) >= 0.) */
        else:
            sign = 1
            dgam = 1 / pow2(xl)  # dgam=1./(xl*xl)
            dgam = sign * ((ct3 * dgam + ct2) * dgam + ct1) / xl
    # % /* if(xlr < tsmag) */
    else:
        dgam = cgam2 - cgam1

    den2 = cksm * dgam / (
    cgam2 * (pow2(k1) * cgam2 + pow2(k2) * cgam1))  # den2=cksm*dgam/(cgam2*(ck1sq*cgam2+ck2sq*cgam1))
    den1 = 1 / (cgam1 + cgam2) - cksm / cgam2
    com = dxl * xl * numpy.exp(-cgam2 * zph)

    answer = numpy.zeros([6], dtype=numpy.complex128)

    answer[5] = com * b0 * den1 / k1  # ans[5] = com*b0*den1/ck1
    com = com * den2  # com *= den2

    if rho != 0:  # if(rho != 0.)
        b0p = b0p / rho
        answer[0] = -com * xl * (b0p + b0 * xl)  # ans[0]=-com*xl*(b0p+b0*xl)
        answer[3] = com * xl * b0p  # ans[3]=com*xl*b0p
    else:
        answer[0] = -com * xl * xl * .5  # ans[0]=-com*xl*xl*.5
        answer[3] = answer[0]  # ans[3]=ans[0]

    answer[1] = com * cgam2 * cgam2 * b0  # ans[1]=com*cgam2*cgam2*b0
    answer[2] = -answer[3] * cgam2 * rho  # ans[2]=-ans[3]*cgam2*rho
    answer[4] = com * b0  # ans[4]=com*b0
    return answer


def evlua(zph, rho, k1, k2):
    # radiating dipole, k
    # receiving dipole, j
    # zph = z_j + z_k
    # rho = radial coordinate for cylindrical system
    # k1 = wave number in slab medium
    # k2 = wave number in top medium

    # int i, jump;
    # static double del, slope, rmis;
    # static complex double cp1, cp2, cp3, bk, delta, delta2, sum[6], ans[6];

    pow2 = misc.power_function(2)

    conj_e = 1
    bk = 0
    # suminc = zeros(6,1);
    answer = numpy.zeros([6])

    tkmag = 100 * numpy.abs(k1)

    delt = zph
    if rho > delt:
        delt = rho

    if zph >= 2 * rho:
        # bessel function form of sommerfeld integrals
        jh = 0
        a = 0
        delt = 1 / delt

        if delt > tkmag:
            b = .1 * (1 - 1j) * tkmag  # b=cmplx(.1*tkmag,-.1*tkmag);
            suminc = rom1(6, 2, zph, rho, k1, k2, a, b, jh)
            a = b
            b = delt * (1 - 1j)
            answer = rom1(6, 2, zph, rho, k1, k2, a, b, jh)
            # for i = 0; i < 6; i++ )
            #    sum[i] += ans[i];
            #    end
            suminc = suminc + answer
        else:
            b = delt * (1 - 1j)
            suminc = rom1(6, 2, zph, rho, k1, k2, a, b, jh)

            delta = .2 * numpy.pi * delt
            answer = gshank(b, delta, answer, 6, suminc, 0, b, b, zph, rho, k1, k2,
                            jh)  # gshank(b,delta,ans,6,sum,0,b,b);
            answer[5] = answer[5] * k1  # ans[5] *= k1;

        if conj_e:
            # conjugate since nec uses exp(+jwt)
            erv = numpy.conj(pow2(k1) * answer[2])  # *erv=conj(ck1sq*ans[2]);
            ezv = numpy.conj(pow2(k1) * (answer[1] + pow2(k2) * answer[4]))  # *ezv=conj(ck1sq*(ans[1]+ck2sq*ans[4]));
            erh = numpy.conj(pow2(k2) * (answer[0] + answer[5]))  # *erh=conj(ck2sq*(ans[0]+ans[5]));
            eph = -numpy.conj(pow2(k2) * (answer[3] + answer[5]))  # *eph=-conj(ck2sq*(ans[3]+ans[5]));
        else:
            # unconjugated
            erv = pow2(k1) * answer[2]  # *erv=conj(ck1sq*ans[2]);
            ezv = pow2(k1) * (answer[1] + pow2(k2) * answer[3])  # *ezv=conj(ck1sq*(ans[1]+ck2sq*ans[4]));
            erh = pow2(k2) * (answer[0] + answer[5])  # *erh=conj(ck2sq*(ans[0]+ans[5]));
            eph = -pow2(k2) * (answer[3] + answer[5])  # *eph=-conj(ck2sq*(ans[3]+ans[5]));

        return erv, ezv, erh, eph
        # # } /* if(zph >= 2.*rho) */
    else:
        # hankel function form of sommerfeld integrals
        jh = 1
        cp1 = .4 * k2 * 1j  # cp1=cmplx(0.0,.4*ck2);
        cp2 = .6 * k2 - .2 * k2 * 1j  # cp2=cmplx(.6*ck2,-.2*ck2);
        cp3 = 1.02 * k2 - .2 * k2 * 1j  # cp3=cmplx(1.02*ck2,-.2*ck2);
        a = cp1
        b = cp2
        suminc = rom1(6, 2, zph, rho, k1, k2, a, b, jh)
        a = cp2
        b = cp3
        answer = rom1(6, 2, zph, rho, k1, k2, a, b, jh)

        # for( i = 0; i < 6; i++ )
        #     sum[i]=-(sum[i]+ans[i]);
        suminc = -(suminc + answer)

        # path from imaginary axis to -infinity
        if zph > .001 * rho:
            slope = rho / zph
        else:
            slope = 1000

        delt = .2 * numpy.pi / delt
        delta = (-1 + slope * 1j) * delt / numpy.sqrt(1 + pow2(slope))
        delta2 = -numpy.conj(delta)
        answer = gshank(cp1, delta, answer, 6, suminc, 0, bk, bk, zph, rho, k1, k2,
                        jh)  # gshank(cp1,delta,ans,6,sum,0,bk,bk);
        rmis = rho * (numpy.real(k1) - k2)

        jump = 0;  # jump = FALSE;
        if (rmis >= 2 * k2) and (rho >= 1E-10):
            if (zph >= 1E-10):
                bk = (-zph + rho * 1j) * (k1 - cp3)  # bk=cmplx(-zph,rho)*(ck1-cp3);
                rmis = -numpy.real(bk) / numpy.abs(numpy.imag(bk))  # rmis=-creal(bk)/fabs(cimag(bk));
                if (rmis > 4 * rho / zph):
                    jump = 1  # jump = TRUE;

            if not jump:  # if( ! jump )
                # integrate up between branch cuts, then to + infinity
                cp1 = k1 - (.1 + .2 * 1j)  # cp1=ck1-(.1+.2fj);
                cp2 = cp1 + .2
                bk = delt * 1j  # bk=cmplx(0.,del);
                suminc = gshank(cp1, bk, suminc, 6, answer, 0, bk, bk, zph, rho, k1, k2,
                                jh)  # gshank(cp1,bk,sum,6,ans,0,bk,bk);
                a = cp1
                b = cp2
                answer = rom1(6, 1, zph, rho, k1, k2, a, b, jh)
                # for( i = 0; i < 6; i++ )
                #     ans[i] -= sum[i];
                answer = answer - suminc

                suminc = gshank(cp3, bk, suminc, 6, answer, 0, bk, bk, zph, rho, k1, k2,
                                jh)  # gshank(cp3,bk,sum,6,ans,0,bk,bk);
                answer = gshank(cp2, delta2, answer, 6, suminc, 0, bk, bk, zph, rho, k1, k2,
                                jh)  # gshank(cp2,delta2,ans,6,sum,0,bk,bk);

            jump = 1  # jump = TRUE;

        # /* if( (rmis >= 2.*ck2) || (rho >= 1.e-10) ) */
        else:
            jump = 0  # jump = FALSE;

        if not jump:  # ( ! jump )
            ##{
            # integrate below branch points, then to + infinity
            #     for( i = 0; i < 6; i++ )
            #       sum[i]=-ans[i];
            suminc = -answer

            rmis = numpy.real(k1) * 1.01  # rmis=creal(ck1)*1.01;
            #     if( (ck2+1.) > rmis )
            #       rmis=ck2+1.;
            if (k2 + 1) > rmis:
                rmis = k2 + 1

            bk = rmis + .99 * numpy.imag(k1) * 1j  # bk=cmplx(rmis,.99*cimag(ck1));
            delta = bk - cp3
            delta = delta * delt / numpy.abs(delta)  # delta *= del/cabs(delta);
            answer = gshank(cp3, delta, answer, 6, suminc, 1, bk, delta2, zph, rho, k1, k2,
                            jh)  # gshank(cp3,delta,ans,6,sum,1,bk,delta2);
            # /* if( ! jump ) */

        answer[5] = answer[5] * k1  # ans[5] *= ck1;

        if conj_e:
            # conjugate since nec uses exp(+jwt)
            erv = numpy.conj(pow2(k1) * answer[2])  # *erv=conj(ck1sq*ans[2]);
            ezv = numpy.conj(pow2(k1) * (answer[1] + pow2(k2) * answer[4]))  # *ezv=conj(ck1sq*(ans[1]+ck2sq*ans[4]));
            erh = numpy.conj(pow2(k2) * (answer[0] + answer[5]))  # *erh=conj(ck2sq*(ans[0]+ans[5]));
            eph = -numpy.conj(pow2(k2) * (answer[3] + answer[5]))  # *eph=-conj(ck2sq*(ans[3]+ans[5]));
        else:
            # unconjugated
            erv = pow2(k1) * answer[2]
            ezv = pow2(k1) * (answer[1] + pow2(k2) * answer[4])
            erh = pow2(k2) * (answer[0] + answer[5])
            eph = -pow2(k2) * (answer[3] + answer[5])

        return erv, ezv, erh, eph


def precalc_Somm(r, k1, k2, use_mex=False):
    # r: dipole coordinates
    # k1: wave number in substrate
    # k2: wave number in upper medium, e.g., air, water etc.

    pow2 = misc.power_function(2)

    # global use_mex
    N = r.shape[0]

    zr = numpy.zeros([N * N, 2])

    ix = 0
    for j in numpy.arange(N):
        # sprintf('precalc zph rho. %d of %d',j,N)
        for k in numpy.arange(N):
            r_j = r[j, :]
            r_k = r[k, :]
            zph = r_j[2] + r_k[2]
            rho = numpy.sqrt(pow2(r_j[0] - r_k[0]) + pow2(r_j[1] - r_k[1]))

            # round to 4 decimal places
            # zph = round2(zph,.0001);
            # rho = round2(rho,.0001);

            zr[ix, :] = numpy.asarray([zph, rho])  # TODO: Get rid of asarray
            ix = ix + 1

    zr0 = zr
    #zr, m, n = numpy.unique(zr, return_index=True, return_inverse=True)
    zr, m, n = misc.unique_rows(zr, return_index=True, return_inverse=True)
    L = zr.shape[0]

    S = numpy.zeros([L, 4], dtype=numpy.complex128)

    for j in numpy.arange(L):
        # sprintf('precalc S. %d of %d',j,L)
        if use_mex:
            raise Exception("Some unknown function")
            # I = somm(numpy.real(k1), numpy.imag(k1),k2,zr[j,1],zr[j,2])
            # IV_rho = I[1] + 1j*I[2]
            # IV_z  = I[3] + 1j*I[4]
            # IH_rho  = I[5] + 1j*I[6]
            # IH_phi  = I[7] + 1j*I[8]
        else:
            IV_rho, IV_z, IH_rho, IH_phi = evlua(zph, rho, k1, k2)

        S[j, :] = [IV_rho, IV_z, IH_rho, IH_phi]

    return S, n


# void test( double f1r, double f2r, double *tr,
#     double f1i, double f2i, double *ti, double dmin )
def test(f1r, f2r, f1i, f2i, dmin):
    den = numpy.abs(f2r)
    tr = numpy.abs(f2i)

    if den < tr:
        den = tr

    if den < dmin:
        den = dmin

    if den < 1E-37:
        tr = 0
        ti = 0
        return tr, ti

    tr = numpy.abs((f1r - f2r) / den)
    ti = numpy.abs((f1i - f2i) / den)

    return tr, ti


# void lambda( double t, complex double *xlam, complex double *dxlam )
# {
#   *dxlam=b-a;
#   *xlam=a+*dxlam*t;
#   return;
# }
def lambd(t, a, b):
    # a = start of integration interval
    # b = ed of integration interval

    dxlam = b - a
    xlam = a + dxlam * t

    return xlam, dxlam
