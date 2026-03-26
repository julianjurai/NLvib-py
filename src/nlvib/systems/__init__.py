"""
Mechanical system classes for NLvib.

Exports
-------
MechanicalSystem
    Base class for all system types.
SingleMassOscillator
    1-DOF Duffing-type oscillator.
ChainOfOscillators
    n-DOF chain of coupled oscillators.
System_with_PolynomialStiffness
    Multi-DOF system with polynomial stiffness nonlinearity.
FE_ElasticRod
    Finite-element uniform elastic (axial) rod.
FE_EulerBernoulliBeam
    Finite-element Euler-Bernoulli beam model.
build_beam_matrices
    Convenience function to build unreduced global beam matrices.
craig_bampton
    Craig-Bampton (fixed-interface) CMS model reduction.
rubin
    Rubin (free-interface) CMS model reduction.
"""

from nlvib.systems.base import MechanicalSystem
from nlvib.systems.cms import craig_bampton, rubin
from nlvib.systems.fe_beam import FE_EulerBernoulliBeam, build_beam_matrices
from nlvib.systems.fe_rod import FE_ElasticRod
from nlvib.systems.oscillators import ChainOfOscillators, SingleMassOscillator
from nlvib.systems.polynomial import System_with_PolynomialStiffness

__all__ = [
    "MechanicalSystem",
    "ChainOfOscillators",
    "FE_ElasticRod",
    "FE_EulerBernoulliBeam",
    "SingleMassOscillator",
    "System_with_PolynomialStiffness",
    "build_beam_matrices",
    "craig_bampton",
    "rubin",
]
