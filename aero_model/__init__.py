"""
Aerodynamic model package.

FILE STRUCTURE
-------------------------------------------------------------------------------
    aero_model/base.py              The catalog: a full description of every
                                    aero model (arguments, accepted values,
                                    sources of error) plus the AeroModelBase
                                    interface.
    aero_model/simple_drag_polar.py SimpleDragPolar

Importing is unchanged from when this was a single aero_model.py module:

    `from aero_model import SimpleDragPolar`

A DATCOM-table-based `AeroModelBase` subclass is planned (see the README's
Future Work) but not implemented yet. See "ADDING A NEW AERO MODEL" at the 
bottom of the catalog in aero_model/base.py.
"""

from .base import AeroModelBase
from .simple_drag_polar import SimpleDragPolar

__all__ = [
    # Interface
    "AeroModelBase",
    # Models
    "SimpleDragPolar",
]

# Name -> class, keyed by each model's class-level `name`. Useful for building
# an aircraft from a config file/dict rather than from Python constructor calls.
AERO_MODEL_TYPES = {
    SimpleDragPolar.name:   SimpleDragPolar,
}
