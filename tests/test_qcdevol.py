from __future__ import print_function   # makes this work for python2 and 3

import unittest

import qcdevol as qcd
import gvar as gv 
import numpy as np 
from scipy.interpolate import approximate_taylor_polynomial as tayl
from numpy import pi
import gvar.powerseries as ps

qcd.HMIN = 0.

def assert_allclose(a,b, rtol=None, atol=None):
    if rtol is None:
        rtol = 1e-9
    if atol is None:
        atol = 1e-9
    return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


qcd.setdefaults(order=50, rtol=1e-10)

class test_qcdevol(unittest.TestCase):

    def test_alpha_decoupling(self):
        " adding and deleting a flavor "
        # compare m=mu with m!=mu
        m0 = 4
        al4 = qcd.Alpha_s(alpha0=0.1, mu0=1, nf=4)
        al5a = al4.add_quark(m=m0)
        m5 = qcd.M_msb(m0=m0, mu0=m0, alpha=al5a)
        al5b = al4.add_quark(m=m5(8), mu=8)
        al5c = al4.add_quark(m=m5)
        mu = [0.1, 1, 10, 1e4]
        assert_allclose(al5a(mu), al5b(mu), rtol=1e-6)
        assert_allclose(al5a(mu), al5c(mu), rtol=1e-6)

        # check that del_quark undoes add quark
        al4a = al5a.del_quark(m=m5('m'), mu=m5('m'))
        al4b = al5b.del_quark(m=m5(8), mu=8)
        al4c = al5c.del_quark(m=m5)
        al4d = al5c.del_quark(m=m0)
        assert_allclose(al4(mu), al4a(mu))
        assert_allclose(al4(mu), al4b(mu))
        assert_allclose(al4b(mu), al4c(mu))
        assert_allclose(al4b(mu), al4d(mu))

    def test_mmsb_decoupling(self):
        " adding and deleting a flavor "
        # compare m=mu with m!=mu
        m0 = 4.
        al5 = qcd.Alpha_s(alpha0=0.1, mu0=1, nf=5)
        m5 = qcd.M_msb(m0=m0, mu0=m0, alpha=al5)
        m4a = m5.del_quark(m=m5)
        m4b = m5.del_quark(m=m5(8), mu=8)
        m4c = m5.del_quark(m=m0)
        assert_allclose(m4a(6), m4b(6), rtol=1e-6)
        assert_allclose(m4a(6), m4c(6))
        m5c = m4c.add_quark()
        assert_allclose(m5(6), m5c(6))

        # check that add_quark is inverse
        m5p = m4a.add_quark(m=m5)
        assert_allclose(m5(5), m5p(5))

        # array m0
        m5v = qcd.M_msb(m0=[m0, 2*m0], mu0=m0, alpha=al5)
        m4v = m5v.del_quark(m=m0, mu=m0)
        assert_allclose(m4v(6)[0], m4c(6))
        m5vc = m4v.add_quark(m=m0)
        assert_allclose(m5vc(6)[0], m5c(6))
        assert_allclose(m5vc(6), m5v(6))

    def test_alpha_evol(self):
        " evolving alpha "
        # compare default and numerical methods
        for s in qcd.SCHEMES:
            al = qcd.Alpha_s(alpha0=0.1, mu0=1, nf=4, scheme=s)
            mu = [0.1, 1, 10, 1e4]
            assert_allclose(al(mu), al.exact(mu))
            # evolution reversible
            al2 = al.clone(mu0=0.005)
            assert_allclose(al2(1), al(1), rtol=2e-7)
            # now with GVars
            al = qcd.Alpha_s(alpha0=gv.gvar('0.100(1)'), mu0=1, nf=4, scheme=s)
            self.assertEqual(str(al(mu)), str(al.exact(mu)))

    def test_scheme(self):
        mu = 0.1
        kargs = dict(alpha0=.1, mu0=1., nf=4)
        almsb = qcd.Alpha_s(scheme='msb', **kargs)
        al = {s:almsb.clone(s) for s in qcd.SCHEMES}
        for s0 in qcd.SCHEMES:
            als0 = al[s0]
            for s in qcd.SCHEMES:
                als = al[s]
                assert_allclose(als0(mu), als(mu, s0))
                assert_allclose(als0(mu), als.exact(mu, s0))

    def test_opz_evol(self):
        " evolving z "
        nf = 4
        for gamma in [None, qcd.GAMMA_MSB(nf)]:
            al = qcd.Alpha_s(alpha0=0.1, mu0=1, nf=nf)
            z = qcd.OPZ(z0=4, mu0=2, alpha=al, gamma=gamma)
            mu = [0.005, 1, 10, 1e4]
            assert_allclose(z(mu), z.exact(mu), rtol=1e-5)
            # evolution reversible
            m2 = qcd.OPZ(z0=z(.005), mu0=.005, alpha=al, gamma=gamma)
            assert_allclose(m2(2), z(2))
            # z0 array
            za = qcd.OPZ(z0=[4., 8.], mu0=2, alpha=al, gamma=gamma)
            assert_allclose(za(2), [z(2), 2 * z(2)])
            assert_allclose(za([3, 6]), [z(3), 2 * z(6)])
            assert_allclose(za(2), za.exact(2))
            assert_allclose(za([3, 6]), za.exact([3, 6]))
            with self.assertRaises(ValueError):
                za([3,4,5.5])
            with self.assertRaises(ValueError):
                za.exact([3,4,5.5])
            # now with GVars
            al = qcd.Alpha_s(alpha0=gv.gvar('0.100(1)'), mu0=1, nf=nf)
            z = qcd.OPZ(z0=gv.gvar('4.00(1)'), mu0=2, alpha=al, gamma=gamma)
            self.assertEqual(str(z(mu)), str(z.exact(mu)))
        
    def test_opz_evol3(self):
        nf = 4
        mu0 = 4.
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=nf, scheme='v')
        gamma = [1.]
        opza = qcd.OPZ(z0=4, mu0=mu0, alpha=al, gamma=gamma)
        c, r = qcd.SCHEMES['v'](nf)
        gamma_msb = [1.] + list(c)
        # following is same as opza but in msb scheme (=> mu -> mu*r)
        opzb = qcd.OPZ(z0=4, mu0=mu0 * r, alpha=al.clone(scheme='msb'), gamma=gamma_msb)
        mu = np.array([0.1, 4., 10, 1e4])
        n = min(len(opza.gamma_msb), len(opzb.gamma_msb))
        assert_allclose(opza.mu0_msb, opzb.mu0_msb)
        assert_allclose(opza.z0, opzb.z0)
        assert_allclose(opza.r, r * opzb.r)
        assert_allclose(opza.gamma_msb[:n], opzb.gamma_msb[:n])
        assert_allclose(opza(mu), opzb(r * mu))

    def test_opz_evol2(self):
        " z evolution vs exact integral "
        al = qcd.Alpha_s(alpha0=0.1, mu0=1, nf=4) 
        z = qcd.OPZ(z0=4., mu0=1, gamma=qcd.GAMMA_MSB(al.nf), alpha=al)
        for mu in [0.005, 1, 10, 1e4]:
            almu = al(mu)
            opzmu = z(mu)
            nbeta = np.arange(len(al.beta_msb))
            ngamm = np.arange(1, len(z.gamma_msb) + 1)
            def g(als):
                return np.sum(z.gamma_msb * als**ngamm)/ als**2 / np.sum(al.beta_msb * als**nbeta)
            integ = gv.ode.integral(g, (al.alpha0_msb, almu), tol=1e-10)
            lnopzratio = np.log(opzmu / z.z0)
            assert_allclose(integ, lnopzratio)

    def test_clone_alpha(self):
        " alpha.clone(..) "
        for s1 in qcd.SCHEMES:
            for s2 in qcd.SCHEMES:
                als1 = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=4, scheme=s1)
                als2 = als1.clone(scheme=s2, mu0=2.)
                assert_allclose(als1(2.), als2.clone(scheme=s1)(2.))
        
    def test_clone_opz(self):
        " alpha.clone(..) "
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=4.)
        z = qcd.OPZ(z0=4., mu0=2., gamma=qcd.GAMMA_MSB(al.nf), alpha=al)
        assert_allclose(z(8.), z.clone()(8.))
        assert_allclose(z(8.), z.clone(4.)(8))

    def test_maxalpha(self):
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=4., maxalpha=None)
        assert_allclose(al.maxalpha, 1.2279912432533255)

    def test_mu0_scheme(self):
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=4., scheme='v')
        self.assertEqual((1.0, 'v'), al.mu0_scheme)

    def test_clone_mmsb(self):
        " m.clone(...) "
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=4.)
        m = qcd.M_msb(m0=4., mu0=2., alpha=al)
        assert_allclose(m(8.), m.clone(mu0=4.)(8.))
        assert_allclose(m(8.), m.clone()(8.))

    def test_mmsb_evol(self):
        " evolving m_msb "
        al = qcd.Alpha_s(alpha0=0.1, mu0=1, nf=4)
        m = qcd.M_msb(m0=4, mu0=2, alpha=al)
        mu = [0.005, 1, 10, 1e4]
        assert_allclose(m(mu), m.exact(mu), rtol=1e-5)
        # evolution reversible
        m2 = qcd.M_msb(m0=m(.005), mu0=.005, alpha=al)
        assert_allclose(m2(2), m(2))
        # array m0
        ma = qcd.M_msb(m0=[4, 6], mu0=2, alpha=al)
        mu = 2 * ma('2 * m')
        assert_allclose(mu, 2 * ma(mu))
        # now with GVars
        al = qcd.Alpha_s(alpha0=gv.gvar('0.100(1)'), mu0=1, nf=4)
        m = qcd.M_msb(m0=gv.gvar('4.00(1)'), mu0=2, alpha=al)
        self.assertEqual(str(m(mu)), str(m.exact(mu)))

    def test_alpha_qed(self):
        Qu = 2/3
        Qd = -1/3
        alqed = 1/100.
        tmp = -alqed * 2 / (4*np.pi) ** 2
        tmp = {
            0:0, 
            1:tmp * (Qu**2),
            2:tmp * (Qu**2 + Qd**2),  
            3:tmp * (Qu**2 + 2 * Qd**2),  
            4:tmp * (2 * Qu**2 + 2 * Qd**2),  
            5:tmp * (2 * Qu**2 + 3 * Qd**2),  
            6:tmp * (3 * Qu**2 + 3 * Qd**2),  
            }
        fac = np.array([1, (6 - 4/3) / (4*np.pi)])
        for nf in range(7):
            al0 = qcd.Alpha_s(alpha0=0.2, mu0=0.1, nf=nf)
            al = qcd.Alpha_s(alpha0=0.2, mu0=0.1, nf=nf, alpha_qed=alqed)
            dbeta = al.beta_msb - al0.beta_msb
            assert_allclose(dbeta[1:3], tmp[nf] * fac)
            assert_allclose(dbeta[:1], [0.])
            assert_allclose(dbeta[3:], len(dbeta[3:]) * [0.])
            if nf > 0:
                self.assertGreater(al(1), al0(1))
            else:
                self.assertEqual(al(1), al0(1))
            # with GVars
            ale = qcd.Alpha_s(alpha0=0.2, mu0=0.1, nf=nf, alpha_qed=alqed * gv.gvar('1.0(1)'))
            assert_allclose(ale(1).mean, al(1))

    def test_msb_qed(self):
        Q = 1/3.
        alqed = 1/100.
        mu0 = 4.
        m0 = 4.
        mu = 8
        gammat = 3 * Q**2 * alqed / (4*np.pi)
        al = qcd.Alpha_s(alpha0=.2, mu0=5., nf=4, alpha_qed=alqed)
        m = qcd.M_msb(m0=m0, mu0=mu0, alpha=al, gamma_msb=[0.], Q=Q)
        assert_allclose(m(mu), m(mu0) * (mu / mu0)**(-2 * gammat))
        m = qcd.M_msb(m0=m0, mu0=mu0, alpha=al, Q=Q)
        m0= qcd.M_msb(m0=m0, mu0=mu0, alpha=al, Q=0)
        dgamma = m.gamma_msb - m0.gamma_msb
        assert_allclose(
            dgamma[:2] / alqed, 
            [4 * Q**2 / (4*np.pi)**2, (-129 + 8*43 - 8/3) * Q**2 / (4*np.pi)**3 ]
            )
        assert_allclose(dgamma[2:], 0)
        mm = m(mu) / (mu / mu0)**(-2 * gammat)
        self.assertGreater(m0(mu), mm)
        assert_allclose(m(mu), m.exact(mu))
        assert_allclose(m0(mu), m0.exact(mu))
        al0 = qcd.Alpha_s(alpha0=.2, mu0=5., nf=4)
        with self.assertRaises(ValueError):
            m = qcd.M_msb(m0=m0, mu0=mu0, alpha=al0, Q=Q)
        # with GVars
        ale = qcd.Alpha_s(alpha0=.2, mu0=5., nf=4, alpha_qed=alqed * gv.gvar('1.0(1)'))
        me = qcd.M_msb(m0=4., mu0=4., alpha=ale, Q=Q)
        assert_allclose(m(8), me(8).mean)

    def test_alpha_ps(self):
        mu0 = 1.
        mu = 0.1
        al = qcd.Alpha_s(alpha0=0.1, mu0=mu0, nf=4, scheme='v')
        for s in qcd.SCHEMES:
            if s == 'V':
                continue
            for s0 in qcd.SCHEMES:
                if s0 == 'V':
                    continue
                als = al.clone(scheme=s)
                ps = al.ps(mu=(mu,s), mu0=(mu0,s0))
                assert_allclose(ps(al(mu0, s0)), al(mu, s))
                if s == 'v' and s0 == 'v':
                    ps = al.ps(mu=mu, mu0=mu0)
                    assert_allclose(ps(al(mu0, s0)), al(mu, s))
        ps = al.ps(mu=(mu,'v'), mu0=(mu,'V'), order=6)
        assert_allclose(ps.c, [0.0, 1.0] + (len(ps.c) - 2) * [0.0])

    def test_opz(self):
        nf = 4
        mu = 0.1
        al = qcd.Alpha_s(0.1, mu0=1., nf=nf)
        gamma = qcd.GAMMA_MSB(al.nf)
        z = qcd.OPZ(z0=12., mu0=5, alpha=al, gamma=gamma)
        m = qcd.M_msb(12., mu0=5., alpha=al)
        assert_allclose(m(mu), z(mu))   

    def test_shape(self):
        al = qcd.Alpha_s(alpha0=.1, mu0=.1, nf=4)
        z = qcd.OPZ(z0=12., mu0=5., gamma=qcd.GAMMA_MSB(al.nf), alpha=al)
        muarray = np.array([[.1], [10.]])
        al1 = al(muarray)
        opz1 = z(muarray)
        al2 = []
        opz2 = []
        for mu in muarray.flat:
            al2.append(al(mu))
            opz2.append(z(mu))
        assert_allclose(al1, np.reshape(al2, muarray.shape))
        assert_allclose(opz1, np.reshape(opz2, muarray.shape))

    def test_msb_str(self):
        " msb('m/10') "
        al = qcd.Alpha_s(alpha0=.1, mu0=1., nf=4)
        m = qcd.M_msb(m0=4, mu0=2., alpha=al)
        m_m_10 = m('m/10')
        assert_allclose(m_m_10, m(m_m_10 / 10))

    def test_custom_beta(self):
        nf = 4
        mu = .1
        m = 4.
        al = qcd.Alpha_s(alpha0=.1, mu0=1., nf=nf)
        alc = qcd.Alpha_s(alpha0=.1, mu0=1., nf=nf, beta_msb=qcd.BETA_MSB(nf, gv.gvar('0(1)')))
        assert_allclose(al(mu), alc(mu).mean)
        self.assertTrue(isinstance(alc(mu), gv.GVar))
        al5 = al.add_quark(m=m, mu=m)
        alc5 = alc.add_quark(m=m, mu=m)
        assert_allclose(al5(mu), alc5(mu).mean)
        al4 = al5.del_quark(m=m, mu=m)
        alc4 = alc5.del_quark(m=m, mu=m)
        assert_allclose(al(mu), al4(mu))
        assert_allclose(al(mu), alc4(mu).mean)
        assert_allclose(al4.beta_msb, np.array(alc4.beta_msb[:-1], dtype=float))

    def test_custom_gamma(self):
        nf = 4 
        m0 = 4.
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=nf)
        m4a = qcd.M_msb(m0=m0, mu0=m0, alpha=al)
        m4b = qcd.M_msb(m0=m0, mu0=m0, alpha=al, gamma_msb=qcd.GAMMA_MSB(nf, gv.gvar('0(1)')))
        assert_allclose(m4a(1.), m4b(1.).mean)
        self.assertTrue(isinstance(m4b(1.), gv.GVar))
        m3a = m4a.del_quark()
        m3b = m4b.del_quark()
        assert_allclose(m3a(1.), m3b(1.).mean)
        self.assertTrue(isinstance(m3b(1.), gv.GVar))
        assert_allclose(m3a.gamma_msb, np.array(m3b.gamma_msb[:-1], dtype=float))
        m4bb = m3b.add_quark(m=m4b)
        assert_allclose(m4a(1), m4bb(1).mean)
        # should be same
        m4bb = m3b.add_quark()
        assert_allclose(m4a(1), m4bb(1).mean)

    def test_custom_zeta2_g(self):
        nf = 4 
        m0 = 4.
        mu = 0.1
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=nf)
        al3 = al.del_quark(m=m0)
        al3c = al.del_quark(m=m0, zeta=qcd.ZETA2_G_MSB(nf=nf, m=m0, mu=m0, terr=gv.gvar('0(1)')))
        assert_allclose(al3(mu), al3c(mu).mean)
        self.assertTrue(isinstance(al3c(mu), gv.GVar))
        al5 = al.add_quark(m=m0)
        al5c = al.add_quark(m=m0, zeta=qcd.ZETA2_G_MSB(nf=nf + 1, m=m0, mu=m0, terr=gv.gvar('0(1)')))
        assert_allclose(al5(mu), al5c(mu).mean)
        self.assertTrue(isinstance(al5c(mu), gv.GVar))

    def test_custom_zeta_m(self):
        nf = 4 
        m0 = 4.
        mu = 0.1
        al = qcd.Alpha_s(alpha0=0.1, mu0=1., nf=nf)
        m = qcd.M_msb(m0=m0, mu0=m0, alpha=al)
        m3 = m.del_quark(m=m0)
        m3c = m.del_quark(m=m0, zeta=qcd.ZETA_M_MSB(nf=nf, m=m0, mu=m0, terr=gv.gvar('0(1)')))
        assert_allclose(m3(mu), m3c(mu).mean)
        self.assertTrue(isinstance(m3c(mu), gv.GVar))
        m5 = m.add_quark(m=m0)
        m5c = m.add_quark(m=m0, zeta=qcd.ZETA_M_MSB(nf=nf+ 1, m=m0, mu=m0, terr=gv.gvar('0(1)')))
        assert_allclose(m5(mu), m5c(mu).mean)
        self.assertTrue(isinstance(m5c(mu), gv.GVar))
        
    def test_exceptions(self):
        " test some exceptions "
        bad = 'x$y$z'
        with self.assertRaises(ValueError):
            qcd.Alpha_s(alpha0=1, mu0=1, nf=4, scheme=bad)
        with self.assertRaises(ValueError):
            qcd.Alpha_s(alpha0=2., mu0=1, nf=4)
        al = qcd.Alpha_s(alpha0=0.2, mu0=5, nf=4)
        with self.assertRaises(ValueError):
            m = qcd.M_msb(m0=1, mu0=1, alpha=al.clone('v'))
        with self.assertRaises(ValueError):
            al(1., bad)
        with self.assertRaises(ValueError):
            al.exact(1., bad)
        with self.assertRaises(ValueError):
            al.ps(mu=10, mu0=(2, bad))
        with self.assertRaises(ValueError):
            al.ps(mu=(10, bad), mu0=2)
        with self.assertRaises(ValueError):
            al.clone(scheme=bad)
        ps = gv.powerseries.PowerSeries([1.,2.], order=5)
        with self.assertRaises(ValueError):
            qcd.evol_ps(ps, mu=2, mu0=(1., bad), nf=4)
        with self.assertRaises(ValueError):
            qcd.evol_ps(ps, mu=(2, bad), mu0=1., nf=4)
        with self.assertRaises(RuntimeError):
            al(1e-6)
        with self.assertRaises(RuntimeError):
            al(0.4)
        with self.assertRaises(RuntimeError):
            al([1e-6])
        with self.assertRaises(RuntimeError):
            al([0.4])

    def test_evol_ps(self):
        nf = 4
        mu0 = 1.
        mu = 0.1 
        m0 = 5.
        order = qcd.ORDER
        almu0_ps = ps.PowerSeries([0, 1], order=order)
        almu_ps = qcd.evol_ps(almu0_ps, mu=mu, mu0=mu0, nf=nf)
        mmu0_ps = ps.PowerSeries([m0], order=order)
        mmu_ps = qcd.evol_ps(mmu0_ps, mu=mu, mu0=mu0, nf=nf, gamma=qcd.GAMMA_MSB(nf))
        # check
        al = qcd.Alpha_s(alpha0=0.1, mu0=mu0, nf=nf)
        assert_allclose(al(mu0), almu_ps(al(mu)))
        m = qcd.M_msb(m0=m0, mu0=mu0, alpha=al)
        assert_allclose(m(mu), mmu_ps(al(mu)))
        # check against evol_xips
        order = 10
        almu0_ps = ps.PowerSeries([0, 1], order=order)
        almu_ps = qcd.evol_ps(almu0_ps, mu=mu, mu0=mu0, nf=nf)
        almu_xips = qcd.evol_xips(almu0_ps, xi=mu, xi0=mu0, nf=nf)
        assert_allclose(almu_xips.c, almu_ps.c)
        mmu0_ps = ps.PowerSeries([m0], order=order)
        mmu_ps = qcd.evol_ps(mmu0_ps, mu=mu, mu0=mu0, nf=nf, gamma=qcd.GAMMA_MSB(nf))
        mmu_xips = qcd.evol_xips(mmu0_ps, xi=mu, xi0=mu0, nf=nf, gamma=qcd.GAMMA_MSB(nf))
        assert_allclose(mmu_xips.c, mmu_ps.c)

    def test_evol_ps_scheme(self):
        " evol_ps schemes "
        nf = 4
        al = gv.powerseries.PowerSeries([0., 1.], order=5)
        for s in qcd.SCHEMES:
            c, r = qcd.SCHEMES[s](nf)
            al_msb = qcd.evol_ps(al, mu=(r, 'msb'), mu0=(1., s), nf=nf)
            al_s = qcd.evol_ps(al_msb, mu=(1.,s), mu0=(r, 'msb'), nf=nf)
            assert_allclose(al_s.c, al.c, atol=1e-9)
            if len(c) == 0:
                assert_allclose(al_msb.c[2:], 0, atol=1e-9)
            else:
                assert_allclose(al_msb.c[2:len(c) + 2], c)

    def test_evol_xips(self):
        " evol_xips "
        def exact(mu_mb, n):
            """ pseudoscalar-pseudoscalar vac polarization (nf=nl=4, arXiv:0907.2117)

            Returns 1/(n-4)th root of series (for N>4) after dividing through by the first 
            coefficient.

            Args:
                mu_mb (float): mu/mb(mu)
                n (int): moment

            Returns: Perturbative coefficients (as (gv.gvar.powerseries) 
                for the n-th moment at mu/mb(mu) = mu_mb and
            """
            Rn_coef = {}
            Rn_coef[4] = { 
                0: [1.3333333333],
                1: [3.1111111111, 0.0],
                2: [-0.14140200429, -6.4814814815, 0.],
                3: [-4.2921768959, 3.5706564992, 13.503086420, 0.0],
                }
            Rn_coef[6] = {
                0: [0.53333333333],
                1: [2.0641975309, 1.0666666667],
                2: [7.3054482939, 1.5909465021, -0.044444444444],
                3: [6.8777326692, -7.7353046271, 0.55054869684, 0.032098765432],
                }
            Rn_coef[8] = {
                0: [0.30476190476],
                1: [1.2117107584, 1.2190476190],
                2: [6.2206842117, 4.3372604350, 1.1682539683],
                3: [16.223749101, 7.3256625778, 4.2523182442, -0.064902998236],
                }
            Rn_coef[10] = {
                0: [0.20317460317],
                1: [0.71275585790, 1.2190476190],
                2: [4.5716739893, 4.8064419249, 2.3873015873],
                3: [15.999289274, 15.323088273, 11.034476526, 1.4589065256],
                }
            if n not in Rn_coef:
                raise ValueError(f'bad n: {n}')
            L = (-2 * np.log(mu_mb)) ** np.arange(5)
            coef = 4 * [0]
            for i in Rn_coef[n]:
                coef[i] = np.sum(L[:i+1] * Rn_coef[n][i]) / np.pi ** i
            if n > 4:
                rn = (gv.powerseries.PowerSeries(coef) / coef[0]) ** (1/(n-4.))
            else:
                rn = gv.powerseries.PowerSeries(coef) / coef[0]
            return rn
        
        gamma = qcd.GAMMA_MSB(nf=4)
        gamma_xi = [-0.5] + list(-gamma)
        for n in [4, 6, 8, 10]:
            ps0 = exact(1, n=n)
            for mu_mb in [.5, 0.67, 1., 1.5, 2.0]:
                ps = qcd.evol_xips(ps0, xi=mu_mb, xi0=1, nf=4, gamma=gamma if n>4 else None, gamma_xi=gamma_xi)
                ex = exact(mu_mb, n)
                assert np.allclose(ps.c, ex.c), f'{n} {ps.c / ex.c}'

    def test_Alphas_Mmsb(self):
        mu = 10.
        m0 = 4.
        mu0 = 4.
        al1 = qcd.Alpha_s(alpha0=.2, mu0=5., nf=4)
        al2 = qcd.Alphas(alpha=.2, mu=5., nf=4)
        assert_allclose(al1(mu), al2(mu))
        assert_allclose(al1.del_quark(m0,mu0)(mu), al2.del_quark(m0,mu0)(mu))
        assert_allclose(al1.add_quark(m0,mu0)(mu), al2.add_quark(m0,mu0)(mu))
        m1 = qcd.M_msb(m0=m0, mu0=mu0, alpha=al1)
        m2 = qcd.Mmsb(m=m0, mu=mu0, alpha=al2)
        assert_allclose(m1(mu), m2(mu))
        assert_allclose(m1.del_quark(m0,mu0)(mu), m2.del_quark(m0,mu0)(mu))
        assert_allclose(m1.add_quark(m0,mu0)(mu), m2.add_quark(m0,mu0)(mu))
        
    def test_alpha_crundec(self):
        alinit = 0.1
        mu = 1.0
        # answers from CRunDec3.1: gpl.cpp
        # should agree to better than al**6 since uses same formalism
        ans = {0:0.0699052473337995, 5:0.0774448968601323}
        for nf in ans:
            al = qcd.Alpha_s(alinit, mu0=mu, nf=nf, scheme='msb')
            assert_allclose(al(10.), ans[nf], rtol=alinit**9, atol=0)

    def test_m_crundec(self):
        alinit = 0.1
        minit = 5.
        mu = 10.
        # answers from CRunDec3.1: gpl.cpp
        # should agree to better than al**6 since uses same formalism
        ans = {0:5.26177001299062, 5:5.24698253625349}
        for nf in ans:
            al = qcd.Alpha_s(alinit, mu0=mu, nf=nf, scheme='msb')
            m = qcd.M_msb(minit, mu0=mu, alpha=al)
            assert_allclose(m('m'), ans[nf], rtol=alinit**9, atol=0)
        ans = {0:4.76782015045946, 5:4.76978957520033}
        for nf in ans:
            al = qcd.Alpha_s(alinit, mu0=mu, nf=nf, scheme='msb')
            m = qcd.M_msb(minit, mu0=mu, alpha=al)
            assert_allclose(m(20), ans[nf], rtol=alinit**9, atol=0)

    def test_alpha_dec_crundec(self):
        alinit = 0.1
        mu = 6.0
        m = 5.0
        # answers from CRunDec3.1: gpl.cpp
        # should agree to better than al**6 since uses same formalism
        ans = {1:0.0998069179241833, 5:0.0998060838815223}
        for nf in ans:
            al = qcd.Alpha_s(alinit, mu0=mu, nf=nf, scheme='msb')
            alm = al.del_quark(5., mu=mu)
            assert_allclose(alm(mu), ans[nf], rtol=alinit**9, atol=0)

    def test_m_dec_crundec(self):
        alinit = 0.1
        mu = 5.0
        mb = 6.0
        ms = 3.0
        # answers from CRunDec3.1: gpl.cpp
        # should agree to better than al**6 since uses same formalism
        ans = {1:3.00105942604084, 5:3.00107076126403}
        for nf in ans:
            al = qcd.Alpha_s(alinit, mu0=mu, nf=nf, scheme='msb')
            m = qcd.M_msb(ms, mu0=mu, alpha=al)
            mm = m.del_quark(mb, mu=mu)
            assert_allclose(mm(mu), ans[nf], rtol=alinit**9, atol=0)

    def test_beta_coef(self):
        " beta function coefficients "
        def old_ref(nf):
            return ([
                (11. - 2./3.*nf) / (4*np.pi),
                (102. - 38./3.*nf) / (4*np.pi)**2,
                (2857/2 - 5033/18*nf + 325/54*nf**2) / (4*np.pi)**3,
                (29242.964 - 6946.2896*nf + 405.08904*nf**2 + 1.499314*nf**3) / (4*np.pi)**4,
                (537147.67 - 186161.95*nf + 17567.758*nf**2 - 231.2777*nf**3 - 1.842474*nf**4)  / (4*np.pi)**5,
            ])
        def ref(nf):
            " from Baikov, Chetyrkin and Kuhn 1502.04719v2  Eq (6)"
            return ([ 
                (2.75 - 0.166667*nf) / pi,
                (6.375 - 0.791667*nf) / pi**2,
                (22.3203 - 4.36892*nf + 0.0940394*nf**2) / pi**3,
                (114.23 - 27.1339*nf + 1.58238*nf**2 + 0.0058567*nf**3) / pi**4,
                (524.56 - 181.8*nf + 17.16*nf**2 - 0.22586*nf**3 - 0.0017993*nf**4) / pi**5
            ])
        for n,deg in zip([0, 1, 2, 3, 4], [1, 1, 2, 3, 4]):
            def f1(nf):
                return [qcd.BETA_MSB(nfi)[n] for nfi in nf]
            def f2(nf):
                return [ref(nfi)[n] for nfi in nf]
            c1 = tayl(f1, x=0, degree=deg, scale=1).c
            c2 = tayl(f2, x=0, degree=deg, scale=1).c
            try:
                assert_allclose(c1, c2, rtol=5e-4)
            except:
                print('xxx term =', n)
                print('BETA_MSB =', c1)
                print('    TEST =', c2)
                print('   RATIO =', c1/c2, '\n')
                raise AssertionError('not close')

    def test_gamma_coef(self):
        " quark mass anomalous dimension coefficients "
        def ref(nf):
            " from Baikov, Chetyrkin and Kuhn 1402.6611v1 Eq (4.1)"
            return ([
                1./np.pi,
                (4.20833 - 0.138889*nf) / (np.pi)**2,
                (19.5156 - 2.28412*nf - 0.0270062*nf**2) / (np.pi)**3,
                (98.9434 - 19.1075*nf + 0.276163*nf**2 + 0.00579322*nf**3) / (np.pi)**4,
                (559.7069 - 143.6864*nf + 7.4824*nf**2 + 0.1083*nf**3 - 0.000085359*nf**4) / (np.pi)**5,
            ])
        for n,deg in zip([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]):
            def f1(nf):
                return [qcd.GAMMA_MSB(nfi)[n] for nfi in nf]
            def f2(nf):
                return [ref(nfi)[n] for nfi in nf]
            c1 = tayl(f1, x=0, degree=deg, scale=1).c
            c2 = tayl(f2, x=0, degree=deg, scale=1).c
            try:
                assert_allclose(c1, c2, rtol=2e-4) #large rtol because ref not accurate for n=4
            except:
                print('xxx  term =', n)
                print('GAMMA_MSB =', c1)
                print('     TEST =', c2)
                print('    RATIO =', c1/c2, '\n')
                raise AssertionError('not close')

    def test_zeta2g_coef(self):
        " alpha_msb quark decoupling coefficients "
        def ref(nf):
            " from Schroder and Steinhauser 0512058 Eq (3.3)"
            return ([
                0., 0.1528/pi**2, (0.9721 - (nf-1)*0.0847)/pi**3,
                (5.1730 - 1.0099*(nf-1) - 0.0220*(nf-1)**2)/pi**4
            ])
        for n,deg in zip([1, 2, 3], [0, 1, 2]):
            def f1(nf_1):
                return [qcd.ZETA2_G_MSB(nf_1i + 1, 1., 1.)[n] for nf_1i in nf_1]
            def f2(nf_1):
                return [ref(nf_1i + 1)[n] for nf_1i in nf_1]
            c1 = tayl(f1, x=0, degree=deg, scale=1).c
            c2 = tayl(f2, x=0, degree=deg, scale=1).c
            try:
                assert_allclose(c1, c2, rtol=1e-3) #large rtol because ref not accurate for n=4
            except:
                print('xxx term =', n)
                print('  ZETA2G =', c1)
                print('    TEST =', c2)
                print('    RATIO =', c1/c2, '\n')
                raise AssertionError('not close')

    def test_zetam_coef(self):
        " m_msb quark decoupling coefficients "
        def ref(nf):
            " from Liu and Steinhauser 1502.04719 Eq (5)"
            return ([
                0.2060/pi**2, (1.848 + 0.02473*(nf-1))/pi**3,
                (6.850 - 1.466*(nf-1) + 0.05616*(nf-1)**2)/pi**4
            ])
        for n,deg in zip([0, 1, 2], [0, 1, 2]):
            def f1(nf_1):
                return [qcd.ZETA_M_MSB(nf_1i + 1, 1., 1.)[n] for nf_1i in nf_1]
            def f2(nf_1):
                return [ref(nf_1i + 1)[n] for nf_1i in nf_1]
            c1 = tayl(f1, x=0, degree=deg, scale=1).c
            c2 = tayl(f2, x=0, degree=deg, scale=1).c
            try:
                assert_allclose(c1, c2, rtol=5e-4) #large rtol because ref not accurate for n=4
            except:
                print('xxx  term =', n)
                print('    ZETAM =', c1)
                print('     TEST =', c2)
                print('    RATIO =', c1/c2, '\n')
                raise AssertionError('not close')

if __name__ == '__main__':
    unittest.main()