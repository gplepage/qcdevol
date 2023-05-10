:mod:`qcdevol` Module
=======================

.. |qcdevol| replace:: :mod:`qcdevol`
.. |gvar| replace:: :class:`gvar.GVar`
.. |msb| replace:: :math:`\overline{\mathrm{MS}}`
.. |Zmu| replace:: :math:`Z(\mu)`
.. |mu| replace:: :math:`\mu`
.. |~| unicode:: U+00A0
    :trim:

.. |alphas| replace:: :math:`\alpha_s`
.. |alphasmu| replace:: :math:`\alpha_s(\mu)`

.. |Alpha_s| replace:: :class:`qcdevol.Alpha_s`
.. |OPZ| replace:: :class:`qcdevol.OPZ`
.. |M_msb| replace:: :class:`qcdevol.M_msb`

.. highlight:: python

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

Introduction
-----------------
.. automodule:: qcdevol
   :synopsis: Evolution  of QCD couplings, masses and other renormalization parameters.

QCD Coupling
--------------
The central  component of the |qcdevol| package is the class that implements 
the QCD coupling constant:

.. autoclass:: qcdevol.Alpha_s 

    |Alpha_s| objects have the  following methods:

        .. automethod:: __call__(mu, scheme=None)

        .. automethod:: del_quark(m, mu=None, zeta=None)

        .. automethod:: add_quark(m, mu=None, zeta=None)

        .. automethod:: clone(scheme=None, mu0=None)

        .. automethod:: ps(mu, mu0, order=None)

        .. automethod:: exact(mu, scheme=None)


Operator Z Factors
--------------------
Multiplying a local operator by its Z factor removes its scale dependence. The 
following class represents such Z factors.

.. autoclass:: qcdevol.OPZ 

    :class:`qcdevol.OPZ` objects have the following methods:

        .. automethod:: __call__(mu)

        .. automethod:: clone(mu0=None)

        .. automethod:: exact(mu)


Running Quark Masses
---------------------
MSbar quark masses are represented by objects of type :class:`qcdevol.M_msb`, 
which are derived from :class:`qcdevol.OPZ`.

.. autoclass:: qcdevol.M_msb 

    :class:`qcdevol.M_msb` objects have the following methods:

        .. automethod:: __call__(mu)

        .. automethod:: del_quark(m=None, mu=None, zeta=None)

        .. automethod:: add_quark(m=None, mu=None, zeta=None)

        .. automethod:: clone(mu0=None)

Functions
------------
The following function is used to evolve and manipulate perturbation 
series:

.. autofunction:: qcdevol.evol_ps 

The default parameters for the beta function, mass 
anomalous dimension, and matching formulas (adding and 
deleting quarks) are obtained from the following functions:

.. autofunction:: qcdevol.BETA_MSB

.. autofunction:: qcdevol.GAMMA_MSB

.. autofunction:: qcdevol.ZETA2_G_MSB 

.. autofunction:: qcdevol.ZETA_M_MSB 

The dictionary ``qcd.SCHEMES`` defines the coupling schemes
available in |qcdevol|. ``qcd.SCHEMES[s](nf)`` returns 
a tuple ``(c,r)`` where the coupling ``als`` in 
scheme ``s`` is related to the MSbar coupling ``almsb``
by::

    als(mu) = almsb(r*mu) * (1 + c[0] * almsb(r*mu) + c[1] * almsb(r*mu)**2 + ...)

The schemes currently available are: ``s='msb'`` for the MSbar 
scheme (default scheme for most functions), and ``s='v'`` or ``s='V'`` 
the static-quark potential coupling, defined by::

    V(Q) = -4 pi C_F alv(Q) / Q**2

through third order in the coupling.

