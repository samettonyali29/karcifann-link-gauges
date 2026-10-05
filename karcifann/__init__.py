"""KarciFANN -- Karci fractional artificial neural networks.

A from-scratch NumPy implementation of the artificial neural network of
Karakurt, Saygili & Karci that replaces the learning rate with the Karci
fractional order derivative::

    W <- W - (J / W)^(alpha - 1) . dJ/dW

Primary references
------------------
* A. Karci, "A new approach for fractional order derivative and its
  applications", Universal J. Engineering Science 1(3), 110-117, 2013.
* A. Karci, "Chain rule for fractional order derivatives",
  Science Innovation 3(6), 63-67, 2015.
* M. Karakurt, H. Saygili, A. Karci, "Karci fractional artificial neural
  networks (KarciFANN): a new artificial neural networks model without
  learning rate and its problems", Turk. J. Elec. Eng. & Comp. Sci. 33(3),
  248-263, 2025.
* M. Karakurt, H. Saygili, A. Karci, "Mathematical model of the KarciFANN
  machine learning method", Firat Univ. J. Eng. Sci. 38(1), 47-56, 2026.
* M. Karakurt, "The effect of weight decay in the KarciFANN method",
  J. Computer Science 10(2), 201-216, 2025.
* M. Caputo, M. Fabrizio, "A new definition of fractional derivative without
  singular kernel", Progr. Fract. Differ. Appl. 1(2), 73-85, 2015.
"""

from .activations import Identity, ReLU, Sigmoid, Softmax, Tanh, TanSig, get_activation
from .caputo_fabrizio import (
    NORMALIZATIONS,
    caputo_fabrizio,
    caputo_factor,
    cf_factor,
    cf_normalization,
    cf_rate,
)
from .datasets import (
    TABULAR,
    digits,
    dry_bean,
    letter,
    load,
    mnist,
    openml_classification,
    optdigits,
    pendigits,
    satimage,
    segment,
    standardize,
    train_val_test_split,
    vehicle,
    xor,
)
from .fod import POWER_MODES, fod_factor, karci_fod, karci_fod_limit
from .gauges import (
    GAUGE_FAMILIES,
    ExpGauge,
    Gauge,
    GaugedDescent,
    LogGauge,
    PowerGauge,
    TanhGauge,
)
from .losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    get_loss,
)
from .audit import EDGE_SPECS, edges_in_rows, find_edge_winners
from .records import RESULTS_DIR, Table
from .metrics import accuracy, confusion_matrix, one_hot, precision_recall_f1
from .network import MLP
from .optimizers import (
    SGD,
    AdaGrad,
    Adam,
    CaputoFabrizioGD,
    CaputoGD,
    KarciFANN,
    Momentum,
    Optimizer,
    RMSProp,
    get_optimizer,
)
from .trainer import History, train

__version__ = "1.0.0"

__all__ = [
    "MLP",
    "KarciFANN",
    "CaputoFabrizioGD",
    "CaputoGD",
    "GaugedDescent",
    "Gauge",
    "PowerGauge",
    "LogGauge",
    "TanhGauge",
    "ExpGauge",
    "GAUGE_FAMILIES",
    "SGD",
    "Momentum",
    "AdaGrad",
    "Adam",
    "RMSProp",
    "Optimizer",
    "get_optimizer",
    "train",
    "History",
    "karci_fod",
    "karci_fod_limit",
    "fod_factor",
    "POWER_MODES",
    "caputo_fabrizio",
    "cf_factor",
    "caputo_factor",
    "cf_normalization",
    "cf_rate",
    "NORMALIZATIONS",
    "Sigmoid",
    "Tanh",
    "TanSig",
    "ReLU",
    "Identity",
    "Softmax",
    "get_activation",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "CategoricalCrossEntropy",
    "BinaryCrossEntropy",
    "get_loss",
    "Table",
    "RESULTS_DIR",
    "find_edge_winners",
    "edges_in_rows",
    "EDGE_SPECS",
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "one_hot",
    "xor",
    "digits",
    "mnist",
    "dry_bean",
    "letter",
    "optdigits",
    "pendigits",
    "satimage",
    "segment",
    "vehicle",
    "openml_classification",
    "TABULAR",
    "load",
    "standardize",
    "train_val_test_split",
    "__version__",
]
