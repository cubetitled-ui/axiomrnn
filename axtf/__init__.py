"""axtf -- the only package that knows about TensorFlow and Keras.

Contains the spiking cell, the Keras layer, and model assembly. If a rule
belongs to training or assembly, it lives here and nowhere else.
"""

__all__ = ["cells", "build"]
