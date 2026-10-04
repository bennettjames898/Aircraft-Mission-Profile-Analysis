"""
Propulsion model package.

FILE STRUCTURE
-------------------------------------------------------------------------------
    propulsion_model/base.py                The catalog: a full description of
                                            every propulsion model (arguments,
                                            accepted values, sources of error)
                                            plus the PropulsionModelBase
                                            interface. READ THIS FIRST.
    propulsion_model/simple_turbofan.py     SimpleTurbofan
    propulsion_model/simple_turboprop.py    SimpleTurboprop

Importing:

    `from propulsion_model import SimpleTurbofan, SimpleTurboprop`

A tabulated, engine-deck-based model is planned (see the README's Future
Work). See "ADDING A NEW PROPULSION MODEL" at the bottom of 
propulsion_model/base.py for expansion instructions.
"""

from .base import PropulsionModelBase
from .simple_turbofan import SimpleTurbofan
from .simple_turboprop import SimpleTurboprop

__all__ = [
    # Interface
    "PropulsionModelBase",
    # Models
    "SimpleTurbofan",
    "SimpleTurboprop",
]

# Name -> class, keyed by each model's class-level `name`. Useful for building
# an aircraft from a config file/dict rather than from Python constructor calls.
PROPULSION_MODEL_TYPES = {
    SimpleTurbofan.name:            SimpleTurbofan,
    SimpleTurboprop.name:           SimpleTurboprop,
}
