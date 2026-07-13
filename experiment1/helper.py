from __future__ import annotations

# Compatibility import layer for existing notebook cells.
# Modules should otherwise be imported by responsibility:
# - experiment_config.py for parameters and training profiles
# - data_pipeline.py for dataset loading and splits
# - model_zoo.py for model classes and parameter counts
# - training_tools.py for metrics, schedulers, and progress helpers
# - training_loop.py for full experiment execution
# - reporting.py for result tables and figures
# - cache_manager.py for saved results, predictions, and checkpoints

from cache_manager import *
from experiment_config import *
from data_pipeline import *
from model_zoo import *
from training_tools import *
from training_loop import *
from reporting import *
