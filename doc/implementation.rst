.. |qcdevol| replace:: :mod:`qcdevol`
.. |gvar| replace:: :class:`gvar.GVar`
.. |msb| replace:: :math:`\overline{\mathrm{MS}}`
.. |Zmu| replace:: :math:`Z(\mu)`
.. |mu| replace:: :math:`\mu`
.. |~| unicode:: U+00A0
    :trim:

.. |alphas| replace:: :math:`\alpha_s`
.. |alphasmu| replace:: :math:`\alpha_s(\mu)`

.. highlight:: python

Implementation
==================

QCD Coupling
--------------
The value of the QCD coupling |alphasmu| at scale |~| |mu| 
is specified by the evolution equation,

.. math::
    \mu^2\frac{d\alpha_s(\mu)}{d\mu^2} = 
    -\alpha_s(\mu)^2\sum_{n=0}^{N_\beta}\beta_n\,\alpha_s^n(\mu),

and an initial value :math:`\alpha_s(\mu_0) = \alpha_0`. 
In |qcdevol|, :math:`N_\beta = 4` for the |msb| scheme; the coefficients
:math:`\beta_n` are given by ``qcdevol.BETA_MSB(nf)``. 

While it is possible 
to integrate the evolution equation directly (see :meth:`Alpha_s.exact`),
a faster method is based upon an expansion of 
the equivalent integrals,

.. math::
    \int\limits_{\mu_0}^{\mu} \frac{d\mu^2}{\mu^2}
    &= - \int\limits_{\alpha_0}^{\alpha_s(\mu)}
    \frac{d\alpha_s}{\alpha_s^2}\,\frac{1}{\sum\limits_{n=0}^{N_\beta}\beta_n\,\alpha_s^n}
    \\
    &\approx - \int\limits_{\alpha_0}^{\alpha_s(\mu)}
    \frac{d\alpha_s}{\beta_0\alpha_s^2}
    \sum\limits_{n=0}^{N_\alpha} d_n\,\alpha_s^n,

where 

.. math::
    d_0 = 1 \quad\quad d_1 = -\beta_1/\beta_0 \quad\quad 
    d_2 = -\beta_2/\beta_0 + \beta_1^2/\beta_0^2
        \quad\quad\cdots

and :math:`N_\alpha` is chosen large enough to give the desired 
accuracy. (By default :math:`N_\alpha=25` in |qcdevol| which is more 
than adequate for most purposes.) Evaluating the integrals and rearranging
gives an equation,

.. math::
    \frac{1}{\alpha_s(\mu)}
    = \frac{1}{\alpha_0} + \beta_0\ln\big(\mu^2/\mu_0^2\big) 
    + d_1 \ln\big(\alpha_s(\mu)/\alpha_0\big)
    + \sum\limits_{n=1}^{N_\alpha-1} \frac{d_{n+1}}{n}\big(\alpha_s^n(\mu) - \alpha_0^n\big),

that determines |alphasmu| implicitly. This equation can be solved for |alphasmu|,
for example, by using a root-finding algorithm like the secant method; typically 
only a few iterations are required.

The radius of convergence for the expansion of the inverse beta function is easily 
calculated. In the |msb| scheme with :math:`n_f=4`, it is :math:`\alpha_\mathrm{max}=1.228`,
which is much larger than what is usually needed.

This approach to calculating the evolution can be efficiently implemented directly
in Python,
without resort to underlying code in |~| C or Cython. This is useful for error 
propagation because it means that any of the parameters can be assigned an 
uncertainty (using the :mod:`gvar` module).

QCD couplings are defined to include a specific number :math:`n_f` of quarks 
in the vacuum polarization. It is possible using perturbation theory  
to remove a heavy quark, with mass :math:`m(\mu)`, from a coupling to 
obtain a new coupling with :math:`n_f-1` flavors:

.. math::
    \alpha_s(\mu, n_f-1) = \alpha_s(\mu,n_f)
    \bigg(1 + \alpha_s(\mu,n_f)
    \sum\limits_{n=0}^{N_{\zeta}^{(\alpha)}}
    \zeta^{(\alpha)}_n \alpha_s^n(\mu,n_f) \bigg)

where the :math:`\zeta^{(\alpha)}_n` depend upon :math:`m(\mu)`, :math:`\mu`, and :math:`n_f`,
and :math:`N_{\zeta}^{(\alpha)}`. The same equation can be used to add a quark 
flavor by solving for :math:`\alpha_s(\mu, n_f)` numerically; adding a quark is then the 
exact inverse of removing a quark.

Quark Mass
-----------------
The evolution of the |msb| mass :math:`m(\mu)` is specified by

.. math::
    \mu^2 \frac{d\ln(m(\mu))}{d\mu^2} = - \alpha_s(\mu)\sum\limits_{n=0}^{N_\gamma}
    \gamma_n\,\alpha_s^n(\mu)

with initial condition :math:`m(\mu_0)=m_0`. In |qcdevol|, :math:`N_\gamma=4`; 
the coefficients :math:`\gamma_n` are given by ``qcdevol.GAMMA_MSB(nf)``.

This equation also can be written as 
an integral 

.. math::
    \int\limits_{m_0}^{m(\mu)} \frac{dm}{m} 
    &= \int\limits_{\alpha_s(\mu_0)}^{\alpha_s(\mu)}\frac{d\alpha_s}{\alpha_s}
    \frac{\sum_{n=0}^{N_\gamma}\gamma_n\,\alpha_s^n}{\sum\limits_{n=0}^{N_\beta} \beta_n\,\alpha_s^n}
    \\
    &\approx \int\limits_{\alpha_s(\mu_0)}^{\alpha_s(\mu)}\frac{d\alpha_s}{\alpha_s}
    \sum\limits_{n=0}^{N_m} \tilde d_n \alpha_s^n

where 

.. math::
    \beta_0 \tilde d_0 = \gamma_0 \quad\quad \beta_0\tilde d_1 = \gamma_1 - \gamma_0\beta_1/\beta_0 
    \quad\quad \cdots

and again :math:`N_m` is chosen large enough to give the desired 
accuracy (:math:`N_m=25` by default). Integrating, we get a closed-form 
result for :math:`m(\mu)` in terms of :math:`m_0` and |alphas|:

.. math::
    \frac{m(\mu)}{m_0} 
    = \bigg(\frac{\alpha_s(\mu)}{\alpha_s(\mu_0)}\bigg)^{\tilde d_0}
    \exp\bigg(\sum\limits_{n=1}^{N_m}\frac{\tilde d_{n}}{n}\big(\alpha_s^n(\mu) - \alpha_s^n(\mu_0)\big)\bigg)

This is very fast to evaluate given values for the coupling.

Heavy quarks can be removed or added to the vacuum polarization assumed in :math:`m(\mu)`, 
analogously to the coupling.


Power Series
-------------
Calculating parameters like the :math:`d_n` above (and manipulating perturbation 
series more generally) is done using the 
:mod:`gvar.powerseries` module:

https://gvar.readthedocs.io/en/latest/gvar_other.html?highlight=powerseries#module-gvar.powerseries