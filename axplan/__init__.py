"""axplan -- budget-aware memory model and partition planner.

TensorFlow-free by construction. Everything here runs without TF, so a
machine with no GPU stack can still plan a model.
"""

__all__ = ["memory", "planner", "credit", "solve"]
