""" 
This package provides tools for evolving QCD couplings, masses and other 
renormalization parameters. Typical usage is::

    >>> import qcdevol as qcd
    >>> al = qcd.Alpha_s(alpha0=0.2128, mu0=5, nf=4)    # coupling
    >>> mc = qcd.M_msb(m0=0.9851, mu0=3, alpha=al)      # c-quark mass
    >>> al(91.2), mc(91.2)                              # run to mu=91.2 GeV
    (0.11269900754817504, 0.6333892033863693)

Uncertainties can be introduced into parameters using the :mod:`gvar` 
module, and these are propagated through the evolution. QED effects 
can be included. There are also tools for manipulating and evolving 
perturbation series (in alpha_s).
"""
#     Copyright (C) 2023 G. Peter Lepage

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

from ._version import __version__
from .qcdevol import *

