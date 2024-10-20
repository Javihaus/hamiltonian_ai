"""
Hamiltonian AI

A library for implementing Hamiltonian-inspired approaches in AI optimization.
"""

# Version of the hamiltonian_ai package
__version__ = "0.1.0"

# Import main components
from .models import *
from .optimizers import *
from .loss_functions import *
from .data_processing import *
from .utils import *

# Define what should be imported with "from hamiltonian_ai import *"
__all__ = [
    'HamiltonianNN',
    'AdvancedSymplecticOptimizer',
    'hamiltonian_loss',
    'HamiltonianDataset',
    'prepare_data',
    'evaluate_model'
]

# Optional: add any initialization code here
def initialize():
    """
    Perform any necessary initialization for the package.
    """
    pass  # Add initialization code if needed

# Optional: Run initialization
initialize()
