"""
OptiWealth Machine Learning Package.

Includes automatic cross-version scikit-learn unpickler shims
to ensure seamless model loading across all scikit-learn versions (1.7 to 1.9+).
"""

import sys
from pathlib import Path

# Add current ml package directory to sys.path
_ML_DIR = str(Path(__file__).resolve().parent)
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

# Scikit-learn 1.7 <-> 1.9+ cross-version loss unpickling compatibility shim
try:
    import sklearn._loss._loss as _sklearn_loss_loss
    if '_loss' not in sys.modules:
        sys.modules['_loss'] = _sklearn_loss_loss
except (ImportError, AttributeError):
    pass

try:
    import sklearn._loss as _sklearn_loss
    if 'sklearn._loss._loss' not in sys.modules:
        sys.modules['sklearn._loss._loss'] = _sklearn_loss
except (ImportError, AttributeError):
    pass
