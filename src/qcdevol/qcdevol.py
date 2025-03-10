# This module implements evolution equations for the coupling constant and 
# operator expectation values in QCD.

#     Copyright (C) 2023-2025 G. Peter Lepage

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings
import numpy
import gvar


RTOL = 1e-10        # relative error allowed in ode integrations
ORDER = 25          # order of implicit expansion
HMIN = 1e-5         # min step size in ode integrations

__all__ = [
    'RTOL', 'ORDER', 'HMIN', 'SCHEMES', 
    'setdefaults', 'Alpha_s', 'OPZ', 'M_msb', 'evol_ps',
    'BETA_MSB', 'GAMMA_MSB', 'ZETA2_G_MSB', 'ZETA_M_MSB',
    'Alphas', 'Mmsb', 'evol_xips'
    ]

# constants used in beta fcns, etc
NU = {0:0, 1:1, 2:1, 3:1, 4:2, 5:2, 6:3}   # 2/3 charge as fcn of nf
ND = {0:0, 1:0, 2:1, 3:2, 4:2, 5:3, 6:3}   # 1/3 charge as fcn of nf
QU = 2. / 3.
QD = -1. / 3.
pi = numpy.pi
exp_m5_6 = numpy.exp(-5./6.)
log_2 = numpy.log(2)
Tf = 0.5
Ca = 3
Cf = 4./3.
Z3 = 1.2020569031595942854  # zeta function
Z2 = pi**2/6.
Z4 = pi**4/90.
Z5 = 1.0369277551433699263
Z6 = 1.0173430619844488 # 1.017343062
Z7 = 1.008349277381923 # 1.008349277
B4 = -1.7628000870737708645
X0 = 1.808879546208334741426364595086952090
A4 = 0.51747906167389934  # polylog(4,1/2)
A5 = 0.50840057924226867  # polylog(5,1/2)

def setdefaults(order=None, rtol=None, hmin=HMIN):
    """
    Set default parameters for |qcdevol|.

    Args:
        order (int): The order in ``alpha_s`` of perturbative expansions 
            used to implement the evolution and matching equations. 
            Increasing ``order`` brings the results closer to what is 
            obtained from integrating the differential evolution equations 
            exactly. Default: 25.
        rtol (positive float): Iterative algorithms used in the evolution 
            and matching equations are stopped after ``order`` iterations 
            or when results have converged to within relative 
            tolerance ``rtol``. Default: 1e-10.
        hmin (positive float): Smallest step size used in numerical 
            integration of the evolution equation (only relevant to 
            exact algorithms).

    Returns:
        Dictionary containing the values of these parameters before they 
        were modified: if ``save = setdefaults(...)`` changes parameters
        then ``setdefaults(**save)`` restores the old values.
    """
    global ORDER, RTOL, HMIN
    old = dict(order=ORDER, rtol=RTOL, hmin=HMIN) 
    if order is not None:
        ORDER = order
    if rtol is not None:
        RTOL = rtol 
    if hmin is not None:
        HMIN = hmin 
    return old

class Alpha_s:
    """
    Running coupling in QCD.

    Typical usage is::

        >>> import qcdevol as qcd
        >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5, nf=4)
        >>> al(5)                   # coupling at 5 GeV
        0.2128
        >>> al(91.2)                # coupling at 91.2 GeV
        0.11269900754817504
        >>> al([5., 91.2])
        array([0.2128    , 0.11269901])

    Args:
        alpha0: Value of coupling at scale ``mu0``. This can be a number, or 
            a variable of type :class:`gvar.GVar` if the value has an 
            uncertainty (see the documentation for :mod:`gvar`).
        mu0: Scale at which ``self(mu0) == alpha0``.
        nf (int): Number of quark flavors included in vacuum polarization.
        scheme (str): Definition/scheme used for the coupling. By default this 
            is either ``'msb'`` for the MSbar scheme or ``'v'`` for the 
            scheme defined in terms of the static quark potential; but new 
            schemes can be added to ``qcdevol.SCHEMES``. Default is ``'msb'``.
        alpha_qed: QED coupling constant. If specified, the leading-order
            QED correction is added to the beta function. The running of 
            the QED coupling is ignored in leading order; ``alpha_qed`` 
            should be chosen to be appropriate for the range of scales 
            of interest. Ignored if ``alpha_qed=None`` (default)
            or if parameter ``beta_msb`` is specified.
        beta_msb (array): Coefficients of the MSbar beta function. If specified
            these are used in place of what is returned by :func:`qcdevol.BETA_MSB`.
            Ignored if ``beta_msb=None`` (default).
        maxalpha (float): Maximum value for alpha allowed in intermediate 
            calculations, which are halted if it is exceeded. Default value 
            is ``maxalpha=1.``. Set ``maxalpha=None`` to have the maximum 
            coupling calculated from the beta function.
        rtol (float): Temporarily replaces the default ``rtol`` parameter; 
            see :func:`qcdevol.setdefaults`.
        order (int): Temporarily replaces the default ``order`` parameter;
            see :func:`qcdevol.setdefaults`.
    """
    def __init__(self, alpha0, mu0, nf, scheme='msb', alpha_qed=None, beta_msb=None, maxalpha=1., rtol=None, order=None):
        self.rtol = RTOL if rtol is None else rtol
        self.order = ORDER if order is None else order
        self.scheme = scheme if scheme is not None else 'msb'
        self._alpha0 = alpha0 
        self._mu0 = mu0
        self.alpha_qed = alpha_qed
        # convert to msb
        if self.scheme != 'msb':
            if self.scheme not in SCHEMES:
                raise ValueError('unkown scheme: {}'.format(self.scheme))
            c, r = SCHEMES[self.scheme](nf)
            self.r = r 
            mu0 = mu0 * self.r
            # calculate alphamsb(r*mu0) corresponding to alpha(mu0) in self.scheme
            if len(c) == 0:
                self.c = numpy.array([0.0])
            else:
                self.c = numpy.asarray(c)
                n = numpy.arange(1, len(self.c) + 1)
                def diff(almsb, alpha0=alpha0):
                    return alpha0 - almsb * (1 + numpy.sum(self.c * almsb**n))
                alpha0 = root_secant(
                    f=diff, x1=alpha0 / (1 + numpy.sum(self.c * alpha0**n)), x2=alpha0,
                    maxit=self.order, rtol=self.rtol, tag='Alpha_s'
                    ) 
        else:
            self.r = 1.
            self.c = numpy.array([0.])              
        # internally use msb values (convert from/to other schemes on input/output)
        self.alpha0_msb = alpha0 
        self.mu0_msb = mu0
        self.nf = nf
        self.beta_msb = (
            BETA_MSB(self.nf, alpha_qed=self.alpha_qed) 
            if beta_msb is None else 
            numpy.asarray(beta_msb)
            )
        self._beta_msb = beta_msb     # save for add_quark, del_quark
        if self.order < len(self.beta_msb) - 1:
            self.order = len(self.beta_msb) - 1
        if maxalpha is None:
            maxalpha = min(numpy.abs(numpy.polynomial.polynomial.Polynomial(self.beta_msb).roots()))
        self.maxalpha = maxalpha
        if self.alpha0_msb > self.maxalpha or self.alpha0_msb <= 0:
            raise ValueError(' bad alpha0_msb (not between 0 and maxalpha): {}'.format(self.alpha0_msb)) 
        self.d = (
            1. / gvar.powerseries.PowerSeries(self.beta_msb / self.beta_msb[0], order=self.order)
            ).c[:]

    def exact(self, mu, scheme=None):
        """ Same as ``Alpha_s.__call__`` except it 
        integrates differential evolution equation directly. It is much slower,
        and is only included for testing purposes. 
        """
        if numpy.shape(mu) != ():
            # raise ValueError('mu arrays not supported')
            mu = numpy.asarray(mu)
            ans = []
            for mu_n in mu.flat:
                ans.append(self.exact(mu_n, scheme))
            return numpy.array(ans).reshape(mu.shape)
        if scheme is not None: 
            if scheme == self.scheme: 
                c, r = self.c, self.r
            elif scheme in SCHEMES:
                c, r = SCHEMES[scheme](self.nf)
            else:
                raise ValueError('unknown scheme: {}'.format(scheme))
            mu = r * mu
        elif self.scheme != 'msb':
            scheme = self.scheme 
            c = self.c 
            mu = self.r * mu 
        else:
            scheme = 'msb'
        delta_t = numpy.log(mu**2 / self.mu0_msb**2)
        n = numpy.arange(len(self.beta_msb))
        def deriv(x, ialpha):
            return delta_t * numpy.sum(self.beta_msb / ialpha ** n)
        odeint = gvar.ode.Integrator(deriv=deriv, tol=self.rtol, hmin=HMIN)
        al = 1. / odeint(1. / self.alpha0_msb, interval=(0., 1.))
        if scheme != 'msb':
            # convert alphamsb to scheme
            n = numpy.arange(1, len(c) + 1)
            al = al * (1 + numpy.sum(c * al**n))
        return al
    
    def __call__(self, mu, scheme=None):
        """ Return the coupling at scale ``mu``, which can be an array.

        If ``scheme`` is not ``None``, the coupling is evaluated in 
        scheme ``scheme``, otherwise the scheme specified when creating 
        the coupling is used. (Default is ``None``)
        """
        efftol = self.rtol ** 0.6     # empirical (= 1/1.618); convergence faster than linear (golden ratio)
        mu_shape = numpy.shape(mu)
        if mu_shape != ():
            mu = numpy.array(mu)
            if mu.dtype == int:
                mu = numpy.array(mu, dtype=float)
        if scheme is not None: 
            if scheme == self.scheme: 
                c, r = self.c, self.r
            elif scheme in SCHEMES:
                c, r = SCHEMES[scheme](self.nf)
            else:
                raise ValueError('unknown scheme: {}'.format(scheme))
            mu_msb = r * mu
        elif self.scheme != 'msb':
            scheme = self.scheme 
            c = self.c 
            mu_msb = self.r * mu 
        else:
            scheme = 'msb'
            mu_msb = mu
        mu_shape = numpy.shape(mu_msb)
        if mu_shape != ():
            mu_msb = numpy.asarray(mu_msb).flat[:]
        nd = len(self.d)
        n = numpy.arange(1, nd - 1)
        d_n = self.d[2:] / n 
        inv_al1 = 1 / self.alpha0_msb + self.beta_msb[0] * numpy.log(mu_msb**2 / self.mu0_msb**2)
        allast = 1 / inv_al1 
        if mu_shape == ():
            if allast <= 0:
                raise RuntimeError('alpha < 0 -- mu too small')
            dn_al0n = d_n * self.alpha0_msb ** n 
            al = 1 / (
                inv_al1 + self.d[1] * numpy.log(allast / self.alpha0_msb)
                + numpy.sum(d_n * allast ** n - dn_al0n)
                )
            if numpy.fabs((al - allast) / al) > efftol:
                difflast = (
                    inv_al1  + self.d[1] * numpy.log(allast / self.alpha0_msb) 
                    + numpy.sum(d_n * allast ** n - dn_al0n) - 1 / allast
                    ) 
                for i in range(len(d_n)):
                    diff = (
                        inv_al1 + self.d[1] * numpy.log(al / self.alpha0_msb) 
                        + numpy.sum(d_n * al ** n - dn_al0n) - 1/al)
                    al, allast = (allast * diff - al * difflast) / (diff - difflast), al 
                    difflast = diff 
                    prec = numpy.fabs((al - allast) / al)
                    if prec < efftol:
                        break
                    if al <= 0:
                        raise RuntimeError('alpha < 0; mu too small')
                    if al > self.maxalpha:
                        raise RuntimeError('alpha > maxalpha: {} > {}'.format(al, self.maxalpha))
            if scheme != 'msb':
                # convert alphamsb to scheme
                n = numpy.arange(1, len(c) + 1)
                al = al * (1 + numpy.sum(c * al**n))
        else:
            if numpy.any(allast <= 0):
                raise RuntimeError('alpha <= 0 -- mu too small')
            n = n[:, None]
            d_n = d_n[:, None]
            dn_al0n = d_n * self.alpha0_msb ** n 
            al = 1 / (
                inv_al1  + self.d[1] * numpy.log(allast / self.alpha0_msb)                
                + numpy.sum(d_n * allast ** n - dn_al0n, axis=0)                
                )
            idx = numpy.fabs((al - allast) / al) > efftol
            if numpy.any(idx):
                difflast = (
                    inv_al1  + self.d[1] * numpy.log(allast / self.alpha0_msb) 
                    + numpy.sum(d_n * allast ** n - dn_al0n, axis=0) - 1 / allast
                    )
                
                for i in range(len(d_n)):
                    diff = (
                        inv_al1 + self.d[1] * numpy.log(al / self.alpha0_msb) 
                        + numpy.sum(d_n * al ** n - dn_al0n, axis=0) - 1/al)
                    altmp = (allast * diff - al * difflast)[idx] / (diff - difflast)[idx]
                    allast = numpy.array(al)
                    al[idx] = altmp
                    difflast = diff 
                    idx = numpy.fabs((al - allast) / al) > efftol
                    if not numpy.any(idx):
                        break
                    if numpy.any(al <= 0):
                        raise RuntimeError('alpha <= 0; mu too small')
                    if numpy.any(al > self.maxalpha):
                        raise RuntimeError('alpha too large, perturbation theory failing: {} > {}'.format(al, self.maxalpha))
            if scheme != 'msb':
                # convert alphamsb to scheme
                n = numpy.arange(1, len(c) + 1)[:, None]
                al = al * (1 + numpy.sum(c[:, None] * al[None, :]**n, axis=0))
            al = al.reshape(mu_shape)
        try:
            self.nit = i
        except:
            self.nit = 0
        return al

    def kargs(self):
        """ Dictionary containing parameters used to create ``self``.
        
        ``qcdevol.Alpha_s(**al.kargs())`` creates a clone of coupling ``al``.
        """
        return dict(
            alpha0=self._alpha0, mu0=self._mu0, nf=self.nf, scheme=self.scheme, 
            alpha_qed=self.alpha_qed, beta_msb=self._beta_msb, maxalpha=self.maxalpha,
            rtol=self.rtol, order=self.order, 
            )    
    
    def get_mu0_scheme(self):
        return self._mu0, self.scheme
    mu0_scheme = property(get_mu0_scheme, doc='tuple containing initial mu0 and scheme')
    
    def clone(self, scheme=None, mu0=None):
        """ Create clone of ``self``.

        This method creates couplings that are equivalent to ``self``
        but potentially in a different scheme or with a different 
        normalization scale ``mu0``.
        
        Args:
            scheme(str or None): Scheme used to define new coupling. 
                Ignored if ``scheme=None`` (default).
            mu0: New normalization scale for coupling. 
                Ignored if ``mu0=None`` (default).

        Returns:
            New object of type :class:`Alpha_s` equivalent to ``self`` 
            but possibly in a different scheme (or with a different 
            ``mu0``).
        """
        _mu0 = mu0
        if scheme is None or scheme == self.scheme:
            mu0 = self.mu0_msb / self.r 
            alpha0 = self(mu0)
            scheme = self.scheme
        elif scheme == 'msb':
            alpha0 = self.alpha0_msb 
            mu0 = self.mu0_msb 
        elif scheme in SCHEMES:
            c, r = SCHEMES[scheme](self.nf)
            n = numpy.arange(1, len(c) + 1)
            alpha0 = self.alpha0_msb * (1 + numpy.sum(c * self.alpha0_msb**n))
            mu0 = self.mu0_msb / r
        else:
            raise ValueError('unknown scheme: {}'.format(scheme))
        kargs = self.kargs()
        kargs['alpha0'] = alpha0 
        kargs['mu0'] = mu0 
        kargs['scheme'] = scheme
        if _mu0 is not None:
            al = Alpha_s(**kargs)
            kargs['alpha0'] = al(_mu0)
            kargs['mu0'] = _mu0    
        return Alpha_s(**kargs)
    
    def del_quark(self, m, mu=None, zeta=None):
        """ Create a new coupling by removing a heavy quark from the vac. polarization.
        
        Creates a new coupling with ``self.nf-1`` flavors by 
        removing a quark whose ``self.nf``-flavor mass is ``m`` 
        at scale ``mu``. If ``mu`` is omitted (or ``None``), 
        it is set equal to ``m``.

        The default ``zeta`` parameters (from 
        ``qcdevol.ZETA2_G_MSB(self.nf, m, mu)``) can be replaced by
        specifying array ``zeta``. Ignored if ``zeta=None`` (default).
        """
        if isinstance(m, M_msb):
            if mu is None:
                mu = m.mu0_msb
            m = m(mu)
        if mu is None:
            mu = m
        alo = self(mu) if self.scheme == 'msb' else self.clone(scheme='msb')(mu)
        if zeta is None:
            zeta = ZETA2_G_MSB(self.nf, m, mu)
        n = numpy.arange(1, len(zeta) + 1)
        aln = alo * (1 + numpy.sum(zeta * alo**n))
        kargs = self.kargs()
        kargs['alpha0'] = aln 
        kargs['mu0'] = mu 
        kargs['nf'] = self.nf - 1
        kargs['scheme'] = 'msb'
        if self._beta_msb is not None:
            beta_msb = numpy.array(self._beta_msb) # make copy
            dbeta = (
                BETA_MSB(self.nf - 1, alpha_qed=self.alpha_qed) 
                - BETA_MSB(self.nf, alpha_qed=self.alpha_qed)
                )
            n = min(len(beta_msb), len(dbeta))
            beta_msb[:n] += dbeta 
            kargs['beta_msb'] = beta_msb
        al = Alpha_s(**kargs)
        return al if self.scheme == 'msb' else al.clone(scheme=self.scheme)
    
    def add_quark(self, m, mu=None, zeta=None):
        """ Create a new coupling by adding a heavy quark to the vac. polarization.
        
        Creates a new coupling with ``self.nf+1`` flavors by 
        adding a quark whose ``(self.nf+1)``-flavor mass is ``m`` 
        at scale ``mu``. If ``mu`` is omitted (or ``None``), 
        it is set equal to ``m``.

        The default ``zeta`` parameters (from 
        ``qcdevol.ZETA2_G_MSB(self.nf+1, m, mu)``) can be replaced by
        specifying array ``zeta``. Ignored if ``zeta=None`` (default).
        """        
        if isinstance(m, M_msb):
            if mu is mu is None:
                mu = m.mu0_msb
            m = m(mu)
        if mu is None:
            mu = m
        alo = self(mu) if self.scheme == 'msb' else self.clone(scheme='msb')(mu)
        if zeta is None:
            zeta = ZETA2_G_MSB(self.nf + 1, m, mu)
        n = numpy.arange(1, len(zeta) + 1)
        def diff(aln):
            return alo - aln * (1 + numpy.sum(zeta * aln**n))
        a2 = alo 
        a1 = alo / (1 + numpy.sum(zeta * alo**n))
        aln = root_secant(diff, x1=a1, x2=a2, maxit=self.order, rtol=self.rtol, tag='Alpha_s.add_quark')
        kargs = self.kargs()
        kargs['alpha0'] = aln 
        kargs['mu0'] = mu 
        kargs['nf'] = self.nf + 1
        kargs['scheme'] = 'msb'
        if self._beta_msb is not None:
            beta_msb = numpy.array(self._beta_msb)   # copy
            dbeta = (
                BETA_MSB(self.nf + 1, alpha_qed=self.alpha_qed) 
                - BETA_MSB(self.nf, alpha_qed=self.alpha_qed)
                )
            n = min(len(beta_msb), len(dbeta))
            beta_msb[:n] += dbeta 
            kargs['beta_msb'] = beta_msb
        al = Alpha_s(**kargs)
        return al if self.scheme == 'msb' else al.clone(scheme=self.scheme)

    def ps(self, mu, mu0=None, order=None):
        """ Power series for ``self(mu)`` in terms of ``self(mu0)``.
        
        Returns a power series ``ps`` of order ``order`` and 
        type :class:`gvar.powerseries.PowerSeries` such that
        such that ``ps(self(mu0))`` is equal to ``self(mu)`` through 
        the order specified.

        Args:
            mu: The expansion gives ``self(mu)``. If ``mu`` is a tuple,
                ``(mu, scheme)``, the expansion is for ``self(*mu)``.
            mu0: The expansion is in powers of ``self(mu0)``. If ``mu0`` 
                is a tuple, ``(mu0,scheme0)``, the expansion is in 
                powers of ``self(*mu0)``.
            order: The order to which the expansion is carried out.
                Leaving ``order=None`` (default) sets ``order`` equal
                to the default set by :func:`qcdevol.setdefaults`.
        """
        # mu0 = (mu0, scheme) or mu0 = mu0 (=> scheme=self.scheme)
        # self(mu) expanded in terms of self(mu0, scheme=scheme0)
        if numpy.shape(mu) != ():
            mu, scheme = mu
        else: 
            scheme = self.scheme 
        alpha = self if scheme == self.scheme else self.clone(scheme=scheme)
        if order is None:
            order = alpha.order
        if order > alpha.order:
            warnings.warn('order disregarded since larger than self.order: {} > {}'.format(order, alpha.order))
            order = alpha.order
        if mu0 is None:
            mu0 = alpha.mu0_msb / alpha.r
            scheme0 = alpha.scheme
        elif numpy.shape(mu0) != ():
            mu0, scheme0 = mu0
        else:
            scheme0 = alpha.scheme
        # convert mu0 to msb if necessary
        if scheme0 != 'msb':
            try:
                c0, r0 = SCHEMES[scheme0](alpha.nf)
            except KeyError:
                raise ValueError('unknown scheme: {}'.format(scheme0))
            mu0 = mu0 * r0
        # convert mu to msb 
        if alpha.scheme != 'msb':
            mu = mu * alpha.r 
        beta0_delta_t = alpha.beta_msb[0] * numpy.log((mu/mu0)**2)
        al0_almu = gvar.powerseries.PowerSeries([1.], order=order)
        al0 = gvar.powerseries.PowerSeries([0, 1.], order=order)
        for i in range(order):
            # order of next 2 statements is important
            almu = al0 / al0_almu
            al0_almu = 1 + al0 * (beta0_delta_t - alpha.d[1] * numpy.log(al0_almu))
            for j in range(1, i + 1): 
                al0_almu += al0 * alpha.d[j + 1] * (almu**j - al0**j) / j
        ps = al0 / al0_almu
        # convert alphamsb(mu) to alpha.scheme
        if alpha.scheme != 'msb':
            ps_n = ps
            tmp = ps
            for cn in alpha.c:
                ps_n *= ps 
                tmp += cn * ps_n 
            ps = tmp 
        if scheme0 != 'msb':
            als = gvar.powerseries.PowerSeries([0, 1.], order=order)
            almsb = gvar.powerseries.PowerSeries([0, 1.], order=order)
            for i in range(order):
                als_almsb = 1
                for j in range(len(c0)):
                    als_almsb += c0[j] * almsb ** (j+1)
                almsb = als / als_almsb 
            ps = ps(almsb)
        return ps 


class OPZ:
    """ Z factor for operator with anomalous dimenion gamma:: 
    
        mu**2 dlog(Z(mu))/dmu**2 = - gammat - alpha(mu) * (gamma[0] + gamma[1] * alpha(mu) + ....)

    Typical usage is::

        >>> import qcdevol as qcd
        >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5., nf=4)
        >>> Z = qcd.OPZ(z0=4., mu0=4., alpha=al, gamma=[0.318, 0.370, 0.321])
        >>> Z(4.)                   # Z at mu = 4 GeV
        4.0
        >>> Z(1000.)                # Z at mu = 1000 GeV
        2.2907519504498515
        >>> z([1., 4., 1000.])
        array([5.74602581, 4.        , 2.38456863])

    Args:
        z0: Value of Z factor at scale ``mu0``. This can be a
            number, or a variable of type :class:`gvar.GVar` if the 
            value has an uncertainty (see the documentation for 
            the :mod:`gvar` module). ``z0`` can also be an 
            array, representing multiple Z factors.
        mu0: Scale at which ``self(mu0)=z0``.
        alpha: QCD coupling (type :class:`qcdevol.Alpha_s`) governing 
            the Z factor's evolution.
        gamma: An array specifying the Z factor's anomalous dimension.
        gammat: Tree-level gamma. Set to zero if ``gammat=None`` (default).
        order (int): Temporarily replaces the default ``order`` parameter; 
            see :func:`qcdevol.setdefaults`.
        rtol (float): Temporarily replaces the default ``rtol`` parameter; 
            see :func:`qcdevol.setdefaults`.
    """
    def __init__(self, z0, mu0, alpha, gamma=None, gammat=None, order=None, rtol=None):
        # if numpy.shape(z0) != ():
        z0 = numpy.array(z0)
        if z0.dtype == int:
            z0 = numpy.array(z0, dtype=float)
        self.z0 = z0 
        if gamma is None:
            gamma = []
        self.gammat = gammat
        self.order = max(ORDER if order is None else order, len(gamma))
        self.scheme = alpha.scheme
        # convert to msb
        if alpha.scheme != 'msb':
            als = alpha.ps(mu=mu0, mu0=(mu0 * alpha.r, 'msb'), order=self.order)
            gamma = gvar.powerseries.PowerSeries([0] + list(gamma), order=self.order)
            gamma = gamma(als).c[1:]
            self.r = alpha.r
            mu0 = mu0 * self.r
            alpha = alpha.clone('msb')
        else:
            self.r = 1.
            gamma = numpy.array(gamma)
        self.mu0_msb = mu0 
        self.alpha_msb = alpha
        self.alpha0_msb = self.alpha_msb(mu0)
        self.gamma_msb = gamma
        num = gvar.powerseries.PowerSeries([0] + list(self.gamma_msb), order=self.order)
        den = gvar.powerseries.PowerSeries(self.alpha_msb.beta_msb, order=self.order)
        self.d = (num / den).c
        self.rtol = RTOL if rtol is None else rtol 

    def kargs(self):
        return dict(
            z0=self.z0, mu0=self.mu0_msb, alpha=self.alpha_msb, 
            gamma=self.gamma_msb, gammat=self.gammat,
            order=self.order, rtol=self.rtol,
            )
    
    def clone(self, mu0=None):
        """ Create clone of ``self``.
        
        If ``mu0`` is not ``None``, parameter ``mu0`` is reset 
        in the clone (with the corresponding value for ``z0``).
        """
        kargs = self.kargs()
        if mu0 is not None:
            kargs['mu0'] = mu0 
            kargs['z0'] = self(mu0)
        return OPZ(**kargs)

    def exact(self, mu):
        """ Same as ``OPZ.__call__`` except it integrates 
        the differential evolution equation directly. It is much slower,
        and is only included for testing purposes. (Particularly
        inefficient if ``self.z0`` and ```mu`` are both arrays.)
        """       
        # ngam = numpy.arange(0, len(self.gamma_msb))
        # nbeta = numpy.arange(2, len(self.alpha_msb.beta_msb) + 2)
        if numpy.shape(mu) != ():
            # raise ValueError('mu arrays not supported')
            mu = numpy.asarray(mu)
            ans = []
            if self.z0.shape == ():
                for mu_n in mu.flat:
                    ans.append(self.exact(mu_n))
            else:
                # very inefficient but used only for testing
                if self.z0.shape != mu.shape:
                    # raise ValueError(f'z0 and mu shapes different: {self.z0.shape} != {mu.shape}')
                    raise ValueError('z0 and mu shapes different: {} != {}'.format(self.z0.shape, mu.shape))
                for i, mu_n in enumerate(mu.flat):
                    ans.append(self.exact(mu_n)[i])
            return numpy.array(ans).reshape(mu.shape)
        mu *= self.r
        ngam = numpy.arange(1, len(self.gamma_msb) + 1)
        nbeta = numpy.arange(2, len(self.alpha_msb.beta_msb) + 2)
        def deriv(als, z):
            return z * numpy.sum(self.gamma_msb * als ** ngam) / numpy.sum(self.alpha_msb.beta_msb * als ** nbeta)
        odeint = gvar.ode.Integrator(deriv, tol=self.rtol, hmin=HMIN)
        if self.gammat is None:
            return odeint(self.z0, (self.alpha0_msb, self.alpha_msb(mu)))
        else:
            fac = (mu / self.mu0_msb) ** (-2 * self.gammat)
            return fac * odeint(self.z0, (self.alpha0_msb, self.alpha_msb(mu)))

    
    def __call__(self, mu):
        """ Return the Z factor at scale ``mu``, which may be an array. 
        
        If ``self.z0`` and ``mu`` are both arrays, they must have the same 
        shape as each element of ``mu`` is matched with the corresponding
        element of ``self.z0``.
        """
        mu_shape = numpy.shape(mu)
        if mu_shape != ():
            mu = numpy.array(mu)
            if mu.dtype == int:
                mu = numpy.array(mu, dtype=float)
            if self.z0.shape != () and self.z0.shape != mu.shape:
                raise ValueError('z0 and mu0 shapes different: {} != {}'.format(self.z0.shape, mu.shape))
        mu *= self.r
        alpha = self.alpha_msb(mu)
        n = numpy.arange(1, len(self.d[2:]) + 1)
        if mu_shape == (): 
            ans = (
                self.z0 * (alpha / self.alpha0_msb) ** self.d[1]
                * numpy.exp(numpy.sum( self.d[2:] / n * (alpha**n - self.alpha0_msb**n) ))
                )
        else:
            n = n[:, None]
            alpha = alpha.flat[:]
            # N.B. numpy automatically converts alpha -> alpha[None, :] in sum so it has right number of indices
            ans = (
                self.z0 * (alpha / self.alpha0_msb) ** self.d[1]
                * numpy.exp(numpy.sum(self.d[2:][:, None] / n * (alpha**n - self.alpha0_msb**n), axis=0))
                ).reshape(mu_shape)
        if self.gammat is None:
            return ans 
        else:
            return ans * (mu / self.mu0_msb) ** (-2 * self.gammat)


class M_msb(OPZ):
    """ MSbar quark masses.
    
    Typical usage::

        >>> import qcdevol as qcd
        >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5., nf=4)
        >>> mc = qcd.M_msb(m0=0.9851, mu0=3., alpha=al)
        >>> mc(3)                   # mc at 3 GeV
        0.9851
        >>> mc(1000.)               # mc at 1000 GeV
        0.5377724915576455
        >>> mc([1., 3., 1000.])
        array([1.43226913, 0.9851    , 0.53777249])

    Args:
        m0: Value of mass at scale mu0. This can be a number, or a 
            variable of type :class:`gvar.GVar` if the value has an 
            uncertainty (see the documentation for the :mod:`gvar` module).
        mu0: Scale at which quark mass equals ``m0``.
        alpha: QCD coupling (type :class:`qcdevol.Alpha_s`) governing 
            the mass's evolution. Must be in the ``'msb'`` scheme.
        gamma_msb (array): Coefficients of the mass's MSbar anomalous 
            dimension. If specified, these are used in place of the 
            values used by :func:`qcdevol.GAMMA_MSB`. Ignored 
            if ``gamma_msb=None`` (default).
        Q: QED charge of quark in units of the proton charge. Ignored 
            if ``None`` (default). When specified, ``alpha`` must 
            also include QED corrections.
        order (int): Temporarily replaces the default ``order`` parameter; 
            see :func:`qcdevol.setdefaults`.
        rtol (float): Temporarily replaces the default ``rtol`` parameter; 
            see :func:`qcdevol.setdefaults`.
    """
    def __init__(self, m0, mu0, alpha, Q=None, gamma_msb=None, order=None, rtol=None):
        if alpha.scheme != 'msb':
            raise ValueError("alpha must be in msb scheme; replace by alpha.clone('msb')")
        alpha_qed = alpha.alpha_qed 
        if Q is not None and alpha_qed is None:
            raise ValueError('alpha must have QED corrections if Q != 0')
        self.Q = Q 
        if alpha_qed is not None:
            if self.Q is None:
                Q = 0.0
            gammat = 3 * Q**2 * alpha_qed / (4 * pi)
            gamma = (
                GAMMA_MSB(alpha.nf, alpha_qed=alpha_qed, Q=Q) 
                if gamma_msb is None else 
                gamma_msb
                )
        else:
            gamma = GAMMA_MSB(alpha.nf) if gamma_msb is None else gamma_msb
            gammat = None
        super().__init__(z0=m0, mu0=mu0, alpha=alpha, gamma=gamma, gammat=gammat, order=order, rtol=rtol)
        self._gamma_msb = gamma_msb  # save for add_quark, del_quark
        self.m0 = self.z0

    def kargs(self):
        return dict(
            m0=self.m0, mu0=self.mu0_msb, alpha=self.alpha_msb, gamma_msb=self._gamma_msb, 
            order=self.order, rtol=self.rtol,
            )
    
    def clone(self, mu0=None):
        """ Create clone of ``self``.
        
        If ``mu0`` is not ``None``, parameter ``mu0`` is reset 
        in the clone (with the corresponding value for ``m0``).
        """
        kargs = self.kargs()
        if mu0 is not None:
            kargs['mu0'] = mu0 
            kargs['m0'] = self(mu0)
        return M_msb(**kargs)
    
    def __call__(self, mu):
        """ Return mass's value at scale ``mu``, which may be an array.
        
        ``mu`` is typically a number but can also be a string containing 
        an arithmetic expression involving parameter ``m``. For example,

            >>> mb = qcd.M_msb(m0=4.513, mu0=3., alpha=al)
            >>> mb('m')             # mu = mb(mu)
            4.231289159789313
            >>> mb('3*m')           # mu = 3*mb(mu)
            3.674028791373564

        calculates ``mb(mu)`` first for ``mu=mb(mu)`` and then for 
        ``mu=mb(3*mu)``.
        """
        if isinstance(mu, str):
            # solve for implicit mu
            mlast = self.z0
            m = self(eval(mu, {'m':mlast}))
            def diff(m):
                return m - self(eval(mu, {'m':m}))
            return root_secant(diff, x1=m, x2=mlast, maxit=self.order, rtol=self.rtol, tag='M_msb')
        else:
            return super().__call__(mu)

    def del_quark(self, m=None, mu=None, zeta=None):
        """ Create a new mass by removing a heavy quark from the vac. polarization.

        Creates a new mass with ``nf-1`` flavors by removing a 
        quark whose ``nf``-flavor mass is ``m`` at scale ``mu``, where ``nf``
        equals ``self.alpha.nf``. If mu is omitted (or None), it is set equal to m.

        If ``m`` is unspecified or ``m=None``, the quark mass is 
        specified by the running mass itself -- that is, it removes 
        itself from the vacuum polarization.

        The default zeta parameters (from qcdevol.ZETA_M_MSB(self.nf, m, mu)) 
        can be replaced by specifying array ``zeta``. Ignored 
        if ``zeta=None`` (default).
        """
        if m is None:
            # deleting self!
            if mu is None:
                mu = self.mu0_msb
            return self.del_quark(m=self(mu), mu=mu)
        if isinstance(m, M_msb):
            if mu is None:
                mu = m.mu0_msb
            m = m(mu)
        if mu is None:
            mu = m
        al = self.alpha_msb(mu)
        if zeta is None:
            zeta = ZETA_M_MSB(nf=self.alpha_msb.nf, m=m, mu=mu)
        gamma_msb = self._gamma_msb 
        if gamma_msb is not None:
            dgamma = GAMMA_MSB(nf=self.alpha_msb.nf - 1) - GAMMA_MSB(nf=self.alpha_msb.nf)
            gamma_msb = numpy.array(gamma_msb) # make copy
            n = min(len(gamma_msb), len(dgamma))
            gamma_msb[:n] += dgamma
        n = numpy.arange(2, len(zeta) + 2)
        new_m = self(mu) * (1 + numpy.sum(zeta * al**n))
        kargs = self.kargs()
        kargs['m0'] = new_m 
        kargs['mu0'] = mu 
        kargs['alpha'] = self.alpha_msb.del_quark(m, mu)
        kargs['gamma_msb'] = gamma_msb
        return M_msb(**kargs)


    def add_quark(self, m=None, mu=None, zeta=None):
        """ Create a new coupling by adding a heavy quark to the vac. polarization.

        Creates a new mass with ``nf+1`` flavors by adding a quark whose 
        ``(nf+1)``-flavor mass is m at scale mu, where ``nf`` equals 
        ``self.alpha.nf``. If mu is omitted (or None), it is set equal to m.

        If ``m`` is unspecified or ``m=None``, the quark mass is 
        specified by the running mass itself -- that is, it adds 
        itself to the vacuum polarization.

        The default zeta parameters (from ``qcdevol.ZETA_M_MSB(nf+1, m, mu)``) 
        can be replaced by specifying array ``zeta``. Ignored 
        if ``zeta=None`` (default).
        """
        if m is None:
            # adding self!
            if mu is None:
                mu = self.mu0_msb
            mlast = self(mu)
            m = self.add_quark(m=mlast, mu=mu)(mu)
            def diff(m):
                return m - self.add_quark(m=m, mu=mu)(mu)
            m = root_secant(f=diff, x1=m, x2=mlast, maxit=self.order, rtol=self.rtol, tag='M_msb.add_quark')
        elif isinstance(m, M_msb):
            if mu is None:
                mu = m.mu0_msb
            m = m(mu)
        if mu is None:
            mu = m
        alpha_new = self.alpha_msb.add_quark(m, mu)
        al = alpha_new(mu)
        if zeta is None:
            zeta = ZETA_M_MSB(nf=alpha_new.nf, m=m, mu=mu)
        gamma_msb = self._gamma_msb
        if gamma_msb is not None: 
            dgamma = GAMMA_MSB(nf=alpha_new.nf) - GAMMA_MSB(nf=alpha_new.nf - 1)
            gamma_msb = numpy.array(gamma_msb) # make copy
            n = min(len(gamma_msb), len(dgamma))
            gamma_msb[:n] += dgamma
        n = numpy.arange(2, len(zeta) + 2)
        new_m = self(mu) / (1 + numpy.sum(zeta * al**n))
        kargs = self.kargs()
        kargs['m0'] = new_m 
        kargs['mu0'] = mu 
        kargs['alpha'] = alpha_new
        kargs['gamma_msb'] = gamma_msb
        return M_msb(**kargs)

def evol_ps(pseries, mu, mu0, nf, gamma=None, beta_msb=None, order=None):
    """ Evolve power series in ``alpha`` from scale ``mu0`` to scale ``mu``.
       
    Given a power series in coupling ``alpha(mu0)`` ::
       
       S(mu0) = sum_n=0..order c_n(mu0) * alpha(mu0)**n
       
    this function calculates the coefficients ``c_n(mu)`` for the 
    the expansion of ``S(mu)`` in powers of ``alpha(mu)`` where ::

        d/dt ln(S(mu))  = - alpha(mu) * sum_n=0..N gamma[n] * alpha(mu)**n

    and ``t=ln(mu**2)``. The new coefficients depend only on the 
    coefficients of the original power series,  the ratio |~| ``mu/mu0``, 
    the number ``nf`` of quark flavors included in the coupling
    and the coefficients ``gamma[n]`` of the anomalous dimension.
    The initial expansion ``S(mu0)`` is specified by ``pseries`` which 
    is an object of type :class:`gvar.powerseries.PowerSeries` 
    (for more information, see the documentation for the :mod:`gvar` 
    module). The final power series ``S(mu)`` is returned as an 
    object of the same type.
    
    Typical usage is ::

        >>> import qcdevol as qcd 
        >>> import gvar as gv
        >>> Z10 = gv.powerseries.PowerSeries([4.], order=5)
        >>> print(Z10.c)            # Z(10) coefficients
        [4. 0. 0. 0. 0. 0.]
        >>> Z12 = qcd.evol_ps(Z10, mu=12., mu0=10., nf=4, gamma=[0.318, 0.370, 0.321])
        >>> print(Z12.c)            # Z(12) coefficients
        [ 4.         -0.46382604 -0.56885921 -0.56718909 -0.18133736  0.34568772]
    
    where ``Z12`` is the expansion at ``mu=12`` of the quantity whose
    expansion at ``mu0=10`` is given by ``Z10`` and whose anomalous dimension
    is specified by ``gamma``.

    Args:
        pseries: Object of type :class:`gvar.powerseries.PowerSeries`  
            representing ``S(mu0)``, a series in powers of ``alpha(mu0)`` 
            with coefficients ``pseries.c[n]``.
        mu: Results returned for the series ``S(mu)`` in terms of ``alpha(mu)``.
            If ``mu`` is tuple ``(mu,scheme)``, ``S(mu)`` is a series in
            powers of ``alpha(mu, scheme)``. The MSbar scheme is assumed
            if ``scheme`` is not specified.
        mu0: Parameter ``pseries`` gives the expansion of ``S(mu0)`` in 
            powers of ``alpha(mu0)``. If ``mu0`` is tuple ``(mu0,scheme0)``.
            ``S(mu0)`` is a series in powers of ``alpha(mu0,scheme0)``.
            The MSbar scheme is assumed if ``scheme0`` is not specified.
        nf (int): Number of quark flavors included in the coupling.
        gamma (array): Coefficients ``gamma[n]`` of the anomalous dimension.
        beta_msb (array or None): If not ``None``, replaces the default 
            MSbar beta function (from :func:`qcdevol.BETA_MSB`). Ignored 
            if ``None`` (default).
        order: Order to which the expansion of ``S(mu)`` is carried. 
            If ``order=None``, ``order`` is set equal to ``pseries.order``.
            Larger values should be chosen for ``order`` if the ratio
            ``mu/mu0`` is large, to capture contributions involving 
            powers of large logarithms in high orders.

    Returns:
        Power series for ``S(mu)`` in powers of ``alpha(mu)``, an object of 
        type :class:`gvar.powerseries.PowerSeries`.
    """
    order = pseries.order if order is None else order # max(pseries.order, order)
    order += 1  # why is this needed?
    pseries = gvar.powerseries.PowerSeries(pseries.c, order=order)
    if numpy.shape(mu) != ():
        mu, scheme = mu
    else:
        scheme = 'msb'
    if numpy.shape(mu0) != ():
        mu0, scheme0 = mu0
    else:
        scheme0 = 'msb' 
    if gamma is None:
        gamma = [0.]
    
    # convert to msb
    if scheme0 != 'msb':
        if scheme0 not in SCHEMES:
            raise ValueError('unknown scheme: {}'.format(scheme0))
        al = Alpha_s(alpha0=.001, mu0=mu0, nf=nf, scheme=scheme0, order=order, beta_msb=beta_msb)
        almsb = al.ps(mu=mu0, mu0=(mu0 * al.r, 'msb'))
        gamma = gvar.powerseries.PowerSeries([0] + list(gamma), order=order)(almsb).c[1:]
        pseries = pseries(almsb)
        r0 = al.r
    else:
        r0 = 1.
    if scheme != 'msb':
        if scheme not in SCHEMES:
            raise ValueError('unkown scheme: {}'.format(scheme))
        c,r = SCHEMES[scheme](nf)
        fac = r * mu / (r0 * mu0)
    else:
        fac = mu / (r0 * mu0)

    almu = Alpha_s(alpha0=.001, mu0=fac, nf=nf, scheme='msb', beta_msb=beta_msb, order=order)
    al0 = almu.ps(1.)
    gamma = numpy.asarray(gamma)
    if numpy.all(gamma == 0):
        ans = pseries(al0)
    else:
        for i0 in range(len(gamma)):
            if gamma[i0] != 0:
                break 
        d = (
            gvar.powerseries.PowerSeries(gamma[i0:] / gamma[i0], order=pseries.order)
            / gvar.powerseries.PowerSeries(almu.beta_msb / almu.beta_msb[0], order=pseries.order)
            ).c[1:]
        almu_ps = gvar.powerseries.PowerSeries([0,1.], order=pseries.order)
        almu_al0 = 1 / gvar.powerseries.PowerSeries(al0.c[1:], order=pseries.order)
        expon = numpy.log(almu_al0) if i0 == 0 else (almu_ps ** i0 - al0 ** i0) / i0
        for j in range(1, len(d) + 1):
            expon += d[j-1] * (almu_ps**(j + i0) - al0**(j + i0)) / (j + i0)
        ans = pseries(al0) * numpy.exp(expon * (gamma[i0] / almu.beta_msb[0]))
    # convert to final scheme
    if scheme != 'msb':
        ans = ans(almu.ps(mu, mu0=(mu / r, scheme)))
    return gvar.powerseries.PowerSeries(ans.c, order=order - 1)

def evol_xips(pseries, xi, xi0, nf, gamma=None, gamma_xi=None, beta_msb=None, order=None, rtol=None):
    """ Evolve power series in ``alpha`` from scale ``mu0`` to scale ``mu``.
       
    Given a power series in coupling ``alpha(mu0)`` ::
       
       S(mu0) = sum_n=0..order c_n(xi(mu0)) * alpha(mu0)**n
       
    this function calculates the coefficients ``c_n(xi(mu))`` for the 
    the expansion of ``S(mu)`` in powers of ``alpha(mu)`` where ::

        d/dt ln(S(mu))  = - alpha(mu) * sum_n=0..N gamma[n] * alpha(mu)**n

        d/dt ln(xi(mu)) = - sum_n=0..M gamma_xi[n] * alpha(mu)**n

    and ``t=ln(mu**2)``. The new coefficients depend only on the 
    coefficients of the original power series,  the ratio |~| ``xi(mu)/xi(mu0)``, 
    the number ``nf`` of quark flavors included in the coupling
    and the coefficients ``gamma[n]`` and ``gamma_xi[n]``.
    The initial expansion ``S(mu0)`` is specified by ``pseries`` which 
    is an object of type :class:`gvar.powerseries.PowerSeries` 
    (for more information, see the documentation for the :mod:`gvar` 
    module). The final power series ``S(mu)`` is returned as an 
    object of the same type.
    
    Typical usage is ::

        >>> import qcdevol as qcd
        >>> import gvar as gv 
        >>> nf = 4
        >>> r_mu0 = gv.powerseries([ 1., 0.61598858, 0.50421341, -0.10263586])
        >>> gamma_r = [0.31830989, 0.3701037, 0.32080731, 0.28029143, 0.36464844]
        >>> gamma_xi = [-0.5] + list(-qcd.GAMMA_MSB(nf))        # xi = mu/m(mu)
        >>> r_mu = qcd.evol_xips(r_mu0, xi=2, xi0=1, nf=nf, gamma=gamma_r, gamma_xi=gamma_xi)
        >>> print(r)
        [1.         0.17471737 0.46105875 0.48103527]    
    
    where ``r_mu0`` is a power series in ``alphas(mu0)`` evaluated 
    at ``xi0 = mu0/m(mu0) = 1`` and ``r_mu`` is the same series 
    but evaluated at ``xi = mu/m(mu) = 2``. The power series are 
    specified by the values of the first four coefficients in the 
    perturbative expansion. The anomalous dimension for this series 
    is specified by ``gamma_r``, while the anomalous dimension for 
    ``xi(mu) = mu/m(mu)``is specified by ``gamma_xi``. (``m(mu)`` is 
    a quark mass.)

    By default, ``xi(mu)=mu`` and ``gamma_xi=[-0.5]`` when ``gamma_xi is 
    not specified. Then ``evol_xips`` becomes functionally equivalent to
    ``evol_ps`` but is typically slower than the latter. ``evol_xips`` 
    can only be used with the MS-bar scheme.

    Args:
        pseries: Object of type :class:`gvar.powerseries.PowerSeries`  
            representing ``S(mu0)``, a series in powers of ``alpha(mu0)`` 
            with coefficients ``pseries.c[n]`` (functions of ``xi(mu)``).
        xi: Results returned for the series ``S(mu)`` in terms of ``alpha(mu)``
            where ``xi`` equals ``xi(mu)``.
        xi0: Parameter ``pseries`` gives the expansion of ``S(mu0)`` in 
            powers of ``alpha(mu0)`` where ``xi0`` equals ``xi(mu0)``.
        nf (int): Number of quark flavors included in the coupling.
        gamma (array): Coefficients ``gamma[n]`` of the anomalous dimension
            associated with ``pseries`` (see above). 
        gamma_xi (array): Coefficients ``gamma_xi[n]`` that specify 
            ``d/dlog(mu^2) xi(mu)``. If ``gamma_xi`` unspecified (or ``None``),
            ``xi(mu)=mu`` and ``gamma=[-0.5]`` (default).
        beta_msb (array or None): If not ``None``, replaces the default 
            MSbar beta function (from :func:`qcdevol.BETA_MSB`). Ignored 
            if ``None`` (default).
        order: Order to which the expansion of ``S(mu)`` is carried. 
            If ``order=None``, ``order`` is set equal to ``pseries.order``.

    Returns:
        Power series for ``S(mu)`` in powers of ``alpha(mu)``, an object of 
        type :class:`gvar.powerseries.PowerSeries`, where ``xi(mu)`` equals 
        parameter ``xi`` specifies ``mu``.
    """
    order = pseries.order if order is None else order
    pseries = gvar.powerseries.PowerSeries(pseries.c, order=order)
    gamma = gvar.powerseries.PowerSeries([0] if gamma is None else [0] + list(gamma), order=order)
    gamma_xi = gvar.powerseries.PowerSeries([-0.5] if gamma_xi is None else gamma_xi, order=order)
    if gamma_xi.c[0] == 0 or abs(gamma_xi.c[0]) < 1e-10 * max(gamma_xi.c ** 2) ** 0.5:
        raise ValueError(f'gamma_xi.c[0] is zero (or too small): {gamma_xi.c[0]}')
    beta = gvar.powerseries.PowerSeries([0,0] + BETA_MSB(nf).tolist() if beta_msb is None else beta_msb, order=order)
    log_xi_xi0 = numpy.log(xi / xi0)
    def dpseries_dlogxi(xi, pseries_c):
        ps = gvar.powerseries.PowerSeries(pseries_c, order=order)
        dps_dal =  gvar.powerseries.PowerSeries(ps.deriv(), order=order)
        drn_dxi = (ps * gamma - dps_dal * beta) / gamma_xi
        return drn_dxi.c * log_xi_xi0
    with warnings.catch_warnings():
        # overly sensitive to hmin
        # warnings.simplefilter('ignore')
        odeint = gvar.ode.Integrator(deriv=dpseries_dlogxi, tol=RTOL if rtol is None else rtol, hmin=HMIN)
        return gvar.powerseries.PowerSeries(odeint(pseries.c, interval=(0, 1)), order=order)

def root_secant(f, x1, x2, maxit, rtol, tag=""):
    """ Find zero of ``f(x)`` starting from extimates ``x1`` (better) and ``x2``. """
    # N.B. efftol should be rtol ** 0.6 approximately for secant rule
    # x1 = best, x2 = second best
    efftol = rtol ** .6
    if numpy.shape(x1) != ():
        idx = numpy.fabs((x1 - x2) / x1) > efftol
        if not numpy.any(idx):
            return x1 
        f2 = f(x2)
        for i in range(maxit):
            f1 = f(x1)
            xtmp = (x2 * f1 - x1 * f2)[idx] / (f1 - f2)[idx]
            x2 = numpy.array(x1)
            x1[idx] = xtmp
            # x1, x2 = (x2 * f1 - x1 * f2) / (f1 - f2), x1 
            f2 = f1
            idx = numpy.fabs((x1 - x2) / x1) > efftol
            if not numpy.any(idx):
                break 
    else:
        if numpy.fabs((x1 - x2) / x1) <= efftol:
            return x1 
        f2 = f(x2)
        for i in range(maxit):
            f1 = f(x1)
            x1, x2 = (x2 * f1 - x1 * f2) / (f1 - f2), x1 
            f2 = f1
            if numpy.fabs((x1 - x2) / x1) <= efftol:
                break 
    return x1 


# MS bar evolution coefficients
def BETA_MSB(nf, terr=None, alpha_qed=None):
    """ Compute ``b[n]`` for the ``nf``-flavor MSbar coupling ``al``::

        mu**2 dal(mu)/dmu**2 = -b[0]*al**2 -b[1]*al**3 -b[2]*al**4 -b[3]*al**5 -b[4]*al**6

    If ``terr`` (truncation error) is not ``None``, a sixth entry is 
    added to ``b`` with ``b[5]`` equal to ``terr`` times the 
    r.m.s. average of the other coefficients. 
    Ignored if ``terr=None`` (default).
    """
    # From van Ritenbergen et al  hep-ph/9701390v1
    # and Chetyrkin et al hep-ph/9708255v2
    # b4 from Herzog, Ruijl, Ueda, Vermaseren and Vogt 1701.01404 (2017).
    b0,b1,b2,b3,b4 = [ 
        11-2.*nf/3.,
        102.-38.*nf/3.,
        2857./2.-5033./18.*nf+325./54.*nf**2,
        ((149753./6.+3564*Z3) - (1078361./162. + 6508./27.*Z3)*nf
        +( 50065./162. + 6472./81.*Z3)*nf**2 + 1093./729.*nf**3),
        (8157455./16. + 621885./2.*Z3 - 88209./2.*Z4 - 288090*Z5
        +nf*(-336460813./1944. - 4811164./81.*Z3 + 33935./6.*Z4 + 1358995./27.*Z5)
        +nf**2*(25960913./1944. + 698531./81.*Z3 - 10526./9.*Z4 - 381760./81.*Z5)
        +nf**3*(-630559./5832. - 48722./243.*Z3 + 1618./27.*Z4 + 460./9.*Z5)
        +nf**4*(1205./2916. - 152./81.*Z3)),
        ]
    ans = numpy.array([b0/(4.*pi), b1/(4*pi)**2, b2/(4*pi)**3, b3/(4*pi)**4, b4/(4*pi)**5])
    if terr is not None:
        ans_avg = numpy.average(gvar.mean(ans) ** 2) ** 0.5 
        ans = numpy.array(ans.tolist() + [ans_avg * terr])
    if alpha_qed is not None:
        dbeta1 = -alpha_qed * (4 * Tf * (NU[nf] * QU**2 + ND[nf] * QD**2)) / (4 * pi) ** 2
        dbeta2 = dbeta1 * (2 * Ca - Cf) / (4 * pi)
        try:
            ans[1:3] += [dbeta1, dbeta2]
        except TypeError:
            ans = list(ans)
            ans[1] += dbeta1 
            ans[2] += dbeta2 
            ans = numpy.array(ans)
    return ans

def GAMMA_MSB(nf, terr=None, alpha_qed=None, Q=None):
    """ Compute ``g[n]`` for ``nf``-flavor MSbar mass::

        mu**2 dlog(m(mu))/dmu**2=  -g[1]*al - g[2]*al**2 - g[3]*al**3 - g[4]*al**4 - g[5]*al**5

    where ``al = alpha_msb(mu)``.

    If ``terr`` (truncation error) is not ``None``, a sixth entry is 
    added to ``g`` with ``g[5]`` equal to ``terr`` times 
    the r.m.s. average of the other coefficients. 
    Ignored if ``terr=None`` (default).
    """
    # Ref: Chetyrkin et al hep-ph/9708255v2 (1997)   
    # and Baikov, Chetyrkin and Kuhn 1402.6611 (2014) 
    g1,g2,g3,g4,g5 = [ 
        1.,
        (202./3.-20.*nf/9.),
        (1249.+(-2216./27.-160.*Z3/3.)*nf-140.*nf**2/81.),
        (4603055./162.+135680.*Z3/27.-8800.*Z5 +
            (-91723./27.-34192.*Z3/9.+880.*Z4+18400.*Z5/9.)*nf+
            (5242./243.+800.*Z3/9-160.*Z4/3.)*nf**2+
            (-332./243.+64.*Z3/27.)*nf**3),
        (99512327./162. + 46402466./243.*Z3 + 96800*Z3**2 - 698126./9.*Z4 - 231757160./243.*Z5 + 242000*Z6 + 412720*Z7
        +nf*(-150736283./1458. - 12538016./81.*Z3 - 75680./9.*Z3**2 + 2038742./27.*Z4 + 49876180./243.*Z5 - 638000./9.*Z6 - 1820000./27.*Z7)
        +nf**2*(1320742./729. + 2010824./243.*Z3 + 46400./27.*Z3**2 - 166300./27.*Z4 - 264040./81.*Z5 + 92000./27.*Z6)
        +nf**3*(91865./1458. + 12848./81.*Z3 + 448./9.*Z4 - 5120./27.*Z5)
        +nf**4*(-260./243. - 320./243.*Z3 + 64./27.*Z4)
        ),
        ]
    ans = numpy.array([g1/pi, g2/16./pi**2, g3/64./pi**3, g4/256./pi**4, g5/1024./pi**5])
    if terr is not None:
        ans_avg = numpy.average(gvar.mean(ans) ** 2) ** 0.5 
        ans = numpy.array(ans.tolist() + [ans_avg * terr])
    if alpha_qed is not None:
        dgam0 = 3 * Cf * Q**2 * alpha_qed / (4*pi) ** 2
        dgam1 = (
            -129/4. * Cf * Ca * Q**2 + 3 * 129/2. * Cf**2 * Q**2 
            - Cf * Tf *(NU[nf] + ND[nf]) * Q**2 
            + Cf * Tf * (-45 + 48 * Z3) * (ND[nf] * QD**2 + NU[nf] * QU**2)
            ) * alpha_qed / (4*pi) ** 3
        try:
            ans[0:2] += [dgam0, dgam1]
        except TypeError:
            ans = list(ans)
            ans[0] += dgam0 
            ans[1] += dgam1
            ans = numpy.array(ans)
    return ans


# schemes for alpha_s
def ALPHAMSB_CR(nf):
    """ ``c,r = ALPHAMSB_CR(nf)`` implies that::
    
            alphas(mu) = alphamsb(r*mu) + c[0] * alphamsb(r*mu)**2 + c[1] * alphamsb(r*mu)**3 + ...
    
    where ``alphas`` is in the MSbar scheme.
    """
    return numpy.array([]), 1.0

def ALPHAV_CR(nf):
    """ ``c,r = ALPHAV_CR(nf)`` implies that:: 
    
            alphas(mu) = alphamsb(r*mu) + c[0] * alphamsb(r*mu)**2 + c[1] * alphamsb(r*mu)**3 + ...
    
    where ``alphas`` is in the static-quark potential scheme.
    """
    # a1,a2 coefficients from Schroder's paper on alpha_V
    pi2 = pi**2
    pi4 = pi**4
    a1 = 31./9.*Ca - 20./9.*Tf*nf
    a2 = ( (4343./162.+4*pi2-pi4/4.+22*Z3/3.)*Ca**2
            - (1798./81.+56.*Z3/3.)*Ca*Tf*nf
            - (55./3.-16.*Z3)*Cf*Tf*nf + 400./81.*Tf**2*nf**2)
    b0 = 11. -2*nf/3.
    b1 = 2*(51.-19.*nf/3.)
    lnx = -5./3.
    ### coefs for al_v = fcn(al_msb)
    c0 = (a1+b0*lnx)/(4*pi)
    c1 = (a2+(b0*lnx)**2+(b1+2*b0*a1)*lnx)/(4*pi)**2
    return numpy.array([c0, c1]), exp_m5_6

SCHEMES = {'msb':ALPHAMSB_CR, 'v':ALPHAV_CR, 'V':ALPHAV_CR}  # 'v' and 'V' the same

# MSbar quark threshold factor coefficients
def ZETA2_G_MSB(nf, m, mu, terr=None):
    """ Compute ``z[i]`` for ``nf`` flavor MSbar ``alpha`` where::

        alpha(mu,nf-1) = z * alpha(mu,nf)

        z = 1 + z[0]*alpha(mu,nf) + z[1]*alpha(mu,nf)**2 + z[2]*alpha(mu,nf)**3 + z[3]*alpha(mu,nf)**4

    Here ``m=mh(mu,nf)`` is the MSbar for the quark flavor being removed.

    If ``terr`` is not none, a fifth entry is added to ``z`` with 
    ``z[4]`` equal to ``terr`` times the r.m.s. average of the other
    coefficients.
    """
    # Ref: Chetyrkin et al hep-ph/9708255v2 (1997)
    # and Schroder and Steinhauser 0512058v1 (2005) 
    ln = numpy.log((mu/m)**2)
    ln2 = ln**2
    ln3 = ln2*ln
    ln4 = ln2*ln2
    z0,z1,z2,z3 = [ 
        -ln/6.,
        11./72.-11.*ln/24.+ln2/36.,
        (564731./124416.-82043.*Z3/27648.-955.*ln/576.+53.*ln2/576
        -ln3/216. + (nf-1.)*(-2633./31104.+67.*ln/576.-ln2/36.)),
        291716893./6123600 + 3031309./1306368*log_2**4 
        - 121./4320*log_2**5 - 3031309./217728*Z2*log_2**2
        + 121./432*Z2*log_2**3 - 2362581983./87091200*Z3
        - 76940219./2177280*Z4 + 2057./576*Z4*log_2
        + 1389./256.*Z5 + 3031309./54432*A4 + 121./36*A5
        - 151369./2177280*X0 
        + ln*(7391699./746496 - 2529743./165888*Z3)
        + ln2*2177./3456 - ln3*1883./10368 + ln4/1296.
        + (nf-1)*(
            - 4770941./2239488 + 685./124416*log_2**4
            - 685./20736*Z2*log_2**2 + 3645913./995328*Z3
            - 541549./165888*Z4 + 115./576*Z5 + 685./5184*A4
            + ln*(-110341./373248 + 110779./82944*Z3)
            - ln2*1483./10368 - ln3*127./5184
            )
        + (nf-1)**2*(
            - 271883./4478976 + 167./5184*Z3  + ln*6865./186624
            - ln2*77./20736 + ln3/324.
            )
        ]
    ans = numpy.array([z0/pi, z1/pi**2, z2/pi**3, z3/pi**4])
    if terr is not None:
        ans_avg = numpy.average(gvar.mean(ans) ** 2) ** 0.5 
        ans = numpy.array(ans.tolist() + [ans_avg * terr])
    return ans

def ZETA_M_MSB(nf, m, mu, terr=None):
    """ Compute ``z[n]`` for ``nf`` flavor MSbar mass ``m`` where::

        m(mu,nf-1) = z * m(mu,nf)

        z = 1 + z[0]*alpha(mu,nf)**2 + z[1]*alpha(mu,nf)**3 + z[2]*alpha(mu,nf)**4

    Here ``m=mh(mu,nf)`` is the MSbar mass for the quark flavor being removed.

    If ``terr`` is not none, a fourth entry is added to ``z`` with 
    ``z[3]`` equal to ``terr`` times the r.m.s. average of the other
    coefficients.
    """
    # Ref: Chetyrkin et al hep-ph/9708255v2 (1997)
    # and Liu + Steinhauser  1502.04719v2 (2015)
    ln = numpy.log((mu/m)**2)
    ln2 = ln**2
    ln3 = ln2*ln
    ln4 = ln2*ln2
    z2,z3,z4 = [ 89./432.-5*ln/36.+ln2/12.,
            (2951./2916.-407.*Z3/864.+5.*Z4/4.-B4/36.+(-311./2592.-5.*Z3/6.)*ln
            +175.*ln2/432.+29.*ln3/216.+(nf-1.)*(1327./11664.-2.*Z3/27.
            -53.*ln/432.-ln3/108.)),
            (131968227029./3292047360 - 1924649./4354560*log_2**4
            + 59./1620*log_2**5 + 1924649./725760*Z2*log_2**2
            - 59./162*Z2*log_2**3 - 353193131./40642560*Z3 
            + 1061./576*Z3**2 + 16187201./580608*Z4 - 725./108*Z4*log_2
            - 59015./1728*Z5 - 3935./432*Z6 - 1924649./181440*A4 - 118./27*A5
            + ln*(
                - 2810855./373248 - 31./216*log_2**4 + 31./36*Z2*log_2**2
                - 373261./27648*Z3 + 4123./288*Z4 + 575./72*Z5 - 31./9*A4
                )
            + ln2*(51163./10368 - 155./48*Z3) + 301./324*ln3 + 305./1152*ln4
            +(nf-1)*(
                - 2261435./746496 + 49./2592*log_2**4 - log_2**5/270. 
                - 49./432*Z2*log_2**2 + Z2/27.*log_2**3 - 1075./1728*Z3
                - 1225./3456*Z4 + 49./72*Z4*log_2 + 497./288*Z5 
                + 49./108*A4 + 4./9*A5
                + ln*(
                    16669./31104 + log_2**4/108 - Z2/18*log_2**2
                    + 221./576*Z3 - 163./144*Z4 + 2./9*A4
                    )
                - ln2*7825./10368 - 23./288*ln3 - 5./144*ln4
                )
            +(nf-1)**2*(
                17671./124416 - 5./864*Z3 - 7./96*Z4 
                + ln*(- 3401./46656 + 7./108*Z3) +31./1296*ln2 + ln4/864
                ))
            ]
    ans = numpy.array([z2/pi**2, z3/pi**3, z4/pi**4])
    if terr is not None:
        ans_avg = numpy.average(gvar.mean(ans) ** 2) ** 0.5 
        ans = numpy.array(ans.tolist() + [ans_avg * terr])
    return ans

# legacy classes

class Alphas(Alpha_s):
    """ Legacy code - do not use. """
    def __init__(self, alpha, mu, nf, scheme='msb', fx=3 * [0.0], tol=None, nitn=None, beta_msb=None,
        hmin=None, f4=None):
        if beta_msb is None and fx[0] != 0:
            beta_msb = BETA_MSB(nf, terr=fx[0])
        self.fx = fx
        super().__init__(alpha0=alpha, mu0=mu, nf=nf, scheme=scheme, rtol=tol, order=nitn, beta_msb=beta_msb)

    def __call__(self, mu, scheme=None, method=None):
        if method == 'ode':
            return super().exact(mu, scheme)
        else:
            return super().__call__(mu, scheme)

    def add_quark(self, m, mu=None):
        if isinstance(m, Mmsb):
            if mu is None:
                mu = m.mu
            m = m(mu)
        if mu is None:
            mu = m
        if self.fx[1] != 0.0:
            zeta = ZETA2_G_MSB(self.nf + 1, m, mu, terr=self.fx[1])
        else:
            zeta = None
        return super().add_quark(m=m, mu=mu, zeta=zeta)

    def del_quark(self, m, mu=None):
        if isinstance(m, Mmsb):
            if mu is None:
                mu = m.mu
            m = m(mu)
        if mu is None:
            mu = m
        if self.fx[1] != 0.0:
            zeta = ZETA2_G_MSB(self.nf, m, mu, terr=self.fx[1])
        else:
            zeta = None
        return super().del_quark(m=m, mu=mu, zeta=zeta)

class Mmsb(M_msb):
    """ Legacy code - do not use. """
    def __init__(
        self, m, mu, alpha, fx=2 * [0.0],
        tol=None, hmin=None, nitn=None, gamma_msb=None, f4=None,
        ):
        if gamma_msb is None and fx[0] != 0:
            gamma = GAMMA_MSB(alpha.nf)
            gamma_avg = numpy.average(gamma ** 2) ** 0.5
            gamma_msb = numpy.array(gamma.tolist() + [gamma_avg * fx[0]])
        super().__init__(m0=m, mu0=mu, alpha=alpha, rtol=tol, order=nitn, gamma_msb=gamma_msb)

    def __call__(self, mu, method=None):
        if method == 'ode':
            return super().exact(mu)
        else:
            return super().__call__(mu)

def oldevol_ps(pseries, fac, nf, gamma=None, beta_msb=None):
    return evol_ps(pseries, mu=fac, mu0=1., nf=nf, gamma=gamma, beta_msb=beta_msb)


# 3-Clause BSD License
# Copyright 2023 G. Peter Lepage 
# Created by G. Peter Lepage (Cornell University) in 2023
#
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above 
#     copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright 
#     notice, this list of conditions and the following disclaimer in 
#     the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its 
#     contributors may be used to endorse or promote products derived 
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS 
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.