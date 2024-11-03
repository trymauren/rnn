import yaml
import numpy as np
from utils.activations import *
from utils import text_processing
from text_processing import WORD_EMBEDDING


def data_constructor(loader: yaml.SafeLoader, node :yaml.nodes.MappingNode) ->\
np.ndarray:
    """Construct an activation function object"""
    parameter_dict = loader.construct_mapping(node)
    if type(parameter_dict["X"]) == str and type(parameter_dict["y"]) == str:
        with open(parameter_dict["X"], "r") as f:
            if parameter_dict["is_text"]:
                X_data = text_processing.read_file(f)
            else:
                X_data = np.genfromtxt(f, dtype=float)
        with open(parameter_dict["X"], "r") as f:
            if parameter_dict["is_text"]:
                y_data = text_processing.read_file(f)
            else:
                y_data = np.genfromtxt(f, dtype=float)
    else:
       pass
    return X_data, y_data

def get_loader():
  """Add constructors to PyYAML loader."""
  loader = yaml.SafeLoader
  loader.add_constructor("!fit", data_constructor)
  return loader