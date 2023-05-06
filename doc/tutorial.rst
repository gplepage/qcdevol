.. |qcdevol| replace:: :mod:`qcdevol`
.. |gvar| replace:: :class:`gvar.GVar`
.. |msb| replace:: :math:`\overline{\mathrm{MS}}`
.. |Zmu| replace:: :math:`Z(\mu)`
.. |mu| replace:: :math:`\mu`
.. |~| unicode:: U+00A0
    :trim:

.. highlight:: python

Overview and Tutorial
======================
This module implements QCD  evolution equations for the coupling, quark 
masses, and other quantities, like matrix elements of local operators, 
that have an anomalous dimension. 
The module also provides tools for manipulating and evolving perturbation series.
The coupling and quark-mass values used in the examples are from lattice QCD (....).


QCD Running Coupling
----------------------
A QCD coupling is created by specifying its value ``alpha0`` at some 
scale ``mu0``, together with the number of ``nf`` of quark flavors included
in the vacuum polarization: for example, ::

    >>> import qcdevol as qcd
    >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5., nf=4)
    >>> al(5)
    0.2128

creates a 4-flavor coupling whose value at ``mu=5`` is 0.2128.
The value of the coupling at another scale ``mu`` is given by ``al(mu)``,
where ``mu`` can be an array::

    >>> al(1)
    0.46677059443925056
    >>> al([1., 5., 1000.])
    array([0.46677059, 0.2128    , 0.08190909])

The initial value ``alpha0`` can be specified with an uncertainty using 
the :mod:`gvar` module::

    >>> import gvar as gv
    >>> al = qcd.Alpha_s(alpha0=gv.gvar('0.2128(25)'), mu0=5., nf=4)
    >>> al([1., 5., 1000.])
    array([0.467(15), 0.2128(25), 0.08191(34)], dtype=object)

Here ``alpha0`` is a Gaussian random variable (type ``gvar.GVar``)
representing 0.2128±0.0025. Such variables
can be manipulated in arithmetic statements just like numbers.
Uncertainties (and correlations) are propagated automatically; 
see the documentation 
for :mod:`gvar` for further information.

Heavy-quark flavors can be added or subtracted to create new couplings: 
for example, ::

    >>> al5 = al.add_quark(m=4, mu=4)
    >>> al5(1000.)
    0.08696(40)
    >>> al3 = al.del_quark(m=4, mu=4)
    >>> al3(1000.)
    0.07745(30)

creates a 5-flavor coupling ``al5`` by adding a quark whose 5-flavor 
mass  is ``m=4`` at ``mu=4``, and a 3-flavor coupling by adding a quark
whose 4-flavor mass is ``m=4`` at ``mu=4``. Note that ``del_quark`` undoes
``add_quark`` and vice versa:

    >>> new_al = al5.del_quark(m=4, mu=4)
    >>> new_al(1000.)
    0.08191(34)
    >>> new_al = al3.add_quark(m=4, mu=4)
    >>> new_al(1000.)
    0.08191(34)

The couplings here are all in the standard |msb| scheme. Other 
definitions/schemes for the coupling are supported (see below).


Running Masses
------------------------------
A running mass in the |msb| scheme is defined by the value ``m0`` of the mass 
at some scale ``mu0``, in addition to the running coupling constant: for 
example, ::

    >>> import qcdevol as qcd
    >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5., nf=4)
    >>> mb = qcd.M_msb(m0=4.513, mu0=3, alpha=al)
    >>> mb(3)
    4.513
    >>> mb(1000)
    2.463676027205009
    >>> mb([1., 5., 1000.])
    array([6.56159841, 4.07833966, 2.46367603])

defines a b-quark mass in a theory with ``nf=4`` light-quark flavors (u,d,s,c),
where the value at scale ``mu`` is given by ``mb(mu)``. 

The scale can be replaced by a string containing an arithmetic expression involving
the mass. For example, ``mb('m')`` chooses a scale such that ``mu=mb(mu)``::

    >>> mb('m')                 # mu = mb(mu)
    4.209031231384643
    >>> mb('2*m')               # mu = 2 * mb(mu)
    3.807878006121742

Again, uncertainties can be introduced by making ``m0`` a Gaussian random variable 
using the :mod:`gvar` module::

    >>> al = qcd.Alpha_s(alpha0=gv.gvar('0.2128(25)'), mu0=5., nf=4)
    >>> mb = qcd.M_msb(m0=gv.gvar('4.513(26)'), mu0=3, alpha=al)
    >>> mb(1000.)
    2.464(21)

where now the uncertainty in ``mb(1000)`` comes from both the coupling and the 
initial mass.

The b-quark masses above are for a theory with ``nf=4`` flavors, which is useful at 
scales below the b-quark mass. At higher scales, however, the b |~| quark should 
be included in the vacuum polarization. A new running mass with ``nf=5`` flavors can 
be created using :meth:`M_msb.add_quark`::

    >>> mb5 = mb.add_quark()    # add b to vac. polarization (nf=5)
    >>> mb5(1000)
    2.422(21)

Again :meth:`M_msb.del_quark` undoes :meth:`M_msb.add_quark`, and vice versa.


Operator Z Factors
--------------------------
The |mu| dependence of a local operator defined in the |msb| scheme is canceled 
by multiplying by a |Zmu| factor where

.. math::
    \mu^2\frac{d\ln{Z(\mu)}}{d\mu^2} = - \alpha_s(\mu)\sum_{n=0}^{N_\gamma} \gamma_n\, \alpha_s^n(\mu)

where the right-hand side of this equation is the operator's anomalous dimension. Such 
functions are represented in |qcdevol| by objects of type :class:`OPZ`: for example, ::

    >>> import qcdevol as qcd
    >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5., nf=4)
    >>> z = qcd.OPZ(z0=2.5, mu0=10., alpha=al, gamma=[1., 0.5, 0.25])
    >>> z(10)
    2.5
    >>> z(1000)
    0.7974943209599848

defines a Z factor ``z(mu)`` where the array ``gamma`` specifies the 
coefficients :math:`\gamma_n` in the evolution equation.


Coupling Schemes
-----------------
By default, |qcdevol| uses the standard |msb| scheme to define 
couplings, but other definitions are supported and it is easy 
to switch between schemes. For example, ::

    >>> import qcdevol as qcd
    >>> alv = qcd.Alpha_s(alpha0=0.2618042866, mu0=5, nf=4, scheme='v')
    >>> alv(5)
    0.26180428659999827    
    >>> alv(5, 'msb')
    0.21280000003610486

defines a coupling in the V scheme but ``alv(5, 'msb')`` gives the value 
of the corresponding |msb| coupling at ``mu=5``. The V scheme is defined 
in terms of the static-quark potential and is particularly 
useful when the renormalization scale is set using the BLM criterion (...ref...).
 
The coupling :math:`\alpha_s` in another scheme is specified in 
terms of :math:`\alpha_\mathrm{\overline{MS}}`:

.. math::
    \alpha_s(\mu) \equiv
    \alpha_\mathrm{\overline{MS}}(r\mu)\Big(1 + \alpha_\mathrm{\overline{MS}}(r\mu)
    \sum_{n=0}^{N_s} c_n\, \alpha_\mathrm{\overline{MS}}^{n}(r\mu)
    \Big)

Schemes are implemented using a dictionary ``qcdevol.SCHEMES`` where ``SCHEMES[scheme](nf)``
is a function call that returns a tuple containing the :math:`c_n` coefficients and the 
rescaling parameter :math:`r` for ``nf`` flavors::

    >>> qcd.SCHEMES['msb'](4)           # MS-bar scheme with nf=4
    (array([], dtype=float64), 1.0)
    >>> qcd.SCHEMES['v'](4)             # V scheme with nf=4
    (array([-0.63661977,  0.98060332]), 0.4345982085070782)

From this output above, the ``nf=4`` V coupling is given by:

.. math::
    \alpha_V(\mu) \equiv \alpha_\mathrm{\overline{MS}}(0.435 \mu) 
    - 0.637 \alpha_\mathrm{\overline{MS}}^2(0.435 \mu)
    + 0.981 \alpha_\mathrm{\overline{MS}}^3(0.435 \mu)

Additional schemes can be added to the dictionary. 

QED Effects
--------------

Power Series
-------------
|qcdevol| has a couple of tools for manipulating perturbation series. These series are 
represented by objects of type :class:`gvar.powerseries.PowerSeries` where, for 
example, ::

    >>> from gvar.powerseries import PowerSeries  
    >>> ps = PowerSeries([1, 0.5, 0.25], order=5)
    >>> ps.c
    array([1.  , 0.5 , 0.25, 0.  , 0.  , 0.  ])
    >>> ps(.1)
    1.0525
    >>> (1 / ps).c
    array([ 1.    , -0.5   ,  0.    ,  0.125 , -0.0625,  0.    ])

creates a power series ``ps`` where ``ps(x)`` is ``1 + 0.5 * x + 0.25 * x**2``. The 
coefficients in the power series are given by array ``ps.c``. Objects of type 
:class:`PowerSeries` can be manipulated in arithmetic expressions (e.g., ``1/ps``); the 
``order`` parameter determines the order to which the results of such manipulations are 
retained (here ``order=5``). See the :mod:`gvar` documentation for more information. 

Coupling evolution can be encoded as a perturbative expansion using the :meth:`Alpha_s.ps`
method::

    >>> import qcdevol as qcd 
    >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5., nf=4)
    >>> al_mu = al.ps(mu=10, mu0=5.)
    >>> al_mu.c
    array([ 0.00000000e+00,  1.00000000e+00, -9.19315001e-01,  3.94494408e-01,
           -2.51122798e-02, -2.95286242e-01,  6.88557903e-01, -6.96944872e-01,
            6.62799259e-02,  9.06342876e-01, -1.81543549e+00,  2.11988983e+00,
           -1.09845078e+00, -1.42954114e+00,  4.49291959e+00, -6.15638652e+00,
            4.21269069e+00,  2.52873248e+00, -1.24163903e+01,  1.99914050e+01,
           -1.74877661e+01, -8.36210407e-01,  3.26707481e+01, -6.30713131e+01,
            6.63081007e+01, -1.80173513e+01])    
    >>> al_mu(al(5.))
    0.17484202878404417
    >>> al(10.)
    0.17484202878432245

Here ``al_mu`` is the perturbative expansion of ``al(10)`` in powers of ``al(5.)``. The 
default order for such expansions is |~| 25; it can be reset using 
:func:`qcdevol.setdefaults`. Although the QCD :math:`\beta` function in |qcdevol| 
includes only 5 |~| terms, we generally want to include many more terms in 
expansions from :meth:`Alpha_s.ps` in order to capture the leading 
logarithms from higher orders. (Note how the size of the coefficients grows with 
the order, because of these logarithms.) This is 
especially true when ``mu`` and ``mu0`` are very different. As shown, the power series
evaluated with argument ``al(5.)`` reproduces the value of ``al(10.)`` to high precision.

Expansions of this sort are useful for re-expressing perturbation series in terms of 
different couplings. For example, assume the power series ``ps`` above is the expansion 
of a physical quantity in terms of ``al(10.)``. We can obtain the power series for that 
same quantity but expressed in terms of ``al(5.)`` from ``ps(al_mu)``::

    >>> ps(al(10.))
    1.0950634481495156
    >>> ps_5 = ps(al_mu)
    >>> ps_5.c 
    array([  1.        ,   0.5       ,  -0.2096575 ,  -0.2624103 ,
             0.39597608,  -0.34153157,   0.24708534,   0.12658372,
            -0.68992001,   0.94619112,  -0.60932994,  -0.3442165 ,
             1.74456656,  -2.85761819,   2.4879741 ,   0.13905648,
            -4.55797342,   8.65938371,  -8.91160139,   2.00821817,
            12.10714048, -27.6086077 ,  32.89875863, -15.31519961,
           -29.39399005,  86.51115021])    
    >>> ps_5(al(5.))
    1.0950634481493524
 
Such manipulations are simplified by using :func:`qcdevol.evol_ps`::

        >>> ps_5 = qcd.evol_ps(ps, mu=5., mu0=10., nf=4, order=25)

gives the same result as above. (Note that ``order=25`` is specified so 
that results match with analysis above; absent this specification the 
order is taken from that of ``ps`` -- i.e., ``order=5``.)

Note that :func:`qcdevol.evol_ps` also works for expansions describing 
quantities with an anomalous dimension (specified by 
parameter ``gamma``)::

    >>> zps = PowerSeries([1., .5, .125], order=25)
    >>> zps(al(1))
    1.2606196456987717
    >>> zps_mu = qcd.evol_ps(z_ps, mu=2, mu0=1, nf=4, gamma=[.3, .1])
    >>> zps_mu.c 
    array([1.00000000e+00, 8.41116917e-02, 1.33399272e-01, 2.99767072e-01,
           5.68779314e-01, 1.17136230e+00, 2.46037314e+00, 5.19810496e+00,
           1.10290278e+01, 2.35302136e+01, 5.04927453e+01, 1.08877565e+02,
           2.35723191e+02, 5.12122316e+02, 1.11602245e+03, 2.43865702e+03,
           5.34168259e+03, 1.17258547e+04, 2.57902639e+04, 5.68239130e+04,
           1.25400917e+05, 2.77142855e+05, 6.13319516e+05, 1.35894881e+06,
           3.01447117e+06, 2.01340283e+07])    
    >>> zps_mu(al(2))
    1.059323660860776
    >>> z = qcd.OPZ(z0=zps(al(1)), mu0=1, alpha=al, gamma=[.3, .1])
    >>> z(2)
    1.0593237119435541    

Here a Z factor at scale ``mu0=1`` is specified in terms of a  power series 
in ``al(1)``. The power series for the Z factor at scale ``mu=2`` in terms 
of ``al(2)`` is given by ``zps_mu``.  The coefficients get quite large in 
``zps_mu`` but ``al(2)**n`` gets smaller faster, so the result for 
``zps_mu(al(2))`` is quite accurate (compare with ``z(2)``). Higher orders
are needed when ``mu/mu0`` becomes large or small.

More About Uncertainties
---------------------------
As discussed above, parameters ``alpha0`` in :class:`Alpha_s` and 'm0' in :class:`M_msb` 
can be specified with uncertainties by replacing numbers with Gaussian random variables 
(objects with a mean and standard deviation
of type |gvar|). In fact almost any parameter in the classes and functions discussed
above can be a |gvar|, and the associated uncertainties and correlations 
are propagated through the various methods. This allows for a  comprehensive analysis of
the impact of such uncertainties on results.

The coupling, for example, typically has uncertainties due to the initial values 
(``alpha0``). One might also worry about errors associated due to the fact that only 
the first five terms of the beta function are included by |qcdevol|.  The impact 
of these errors on the coupling value at, say, the Z mass is easily measured by 
running the following code::

    import numpy as np
    import gvar as gv 
    import qcdevol as qcd 

    # uncertainty due to initial value
    alpha0 = gv.gvar('0.2128(25)')
    print('alpha0 =', alpha0)

    # uncertainty in mass of Z
    Mz = gv.gvar('91.1876(21)')
    print('Mz =', Mz, '\n')

    # uncertainty due to higher-order term in beta function
    bmsb = qcd.BETA_MSB(nf=4)
    bmsb_rms = np.average(bmsb ** 2) ** 0.5
    bmsb = np.concatenate((bmsb, [bmsb_rms * gv.gvar('0(1)')]))
    print('extended beta_msb =', bmsb, '\n')

    # create alpha with alph0 and extended beta_msb
    al = qcd.Alpha_s(alpha0=alpha0, mu0=5., nf=4, beta_msb=bmsb)
    print('al(Mz) =', al(Mz), '\n')

    # create error budget for al(Mz)
    inputs = dict(alpha0=alpha0, beta=bmsb[-1], Mz=Mz)
    outputs = {'al(Mz)':al(Mz)}
    print(gv.fmt_errorbudget(inputs=inputs, outputs=outputs, ndecimal=4))

In addition to the uncertainty in ``alpha0``, we include uncertainty in 
the Z |~| mass, and we add a extra term 0.00±0.38 to the beta function, 
beyond the ones 
normally used by |qcdevol| (given by ``qcdevol.BETA_MSB(4)``). 
Running the 
code gives the following ouput::

    alpha0 = 0.2128(25)
    Mz = 91.1876(21) 
    
    extended beta_msb = [0.6631455962162306 0.32507213085250036 0.20477298034366223
     0.32222297343221057 0.1860792072936385 0.00(38)] 
    
    al(Mz) = 0.11270(66) 
    
    Partial % Errors:
                  al(Mz)
    --------------------
       alpha0:    0.5869
         beta:    0.0027
           Mz:    0.0004
    --------------------
        total:    0.5870
    
This shows that ``al(Mz)`` has an uncertainty of 0.5870%, with 0.5869% coming from ``alpha0``,
0.0027% from the extended beta function, and 0.0004% from the Z |~| mass. (The separate
errors are added in quadrature to obtain the total error.) Truncating the 
beta function at five terms has negligible impact on the final error; 
and the uncertainty in the Z |~| mass is even   less important.

Note that the three lines of code 
defining ``bmsb`` could have been collapsed to the single statement ::

    bmsb = qcd.BETA_MSB(4, gv.gvar('0(1)'))
