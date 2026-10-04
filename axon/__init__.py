"""axon -- numpy reference kernel plus the gate.

The reference exists so gradients can be checked against finite differences
without a GPU. gate.py is mandatory before publishing any result.
"""

__all__ = ["gate", "lif"]
