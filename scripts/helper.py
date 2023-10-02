"""Some helper functions."""
import inspect

import numpy as np
import pandas as pd


def print_shapes(*args: pd.DataFrame | np.ndarray) -> None:
    """
    Displays the shape of each object passed as an argument,
    along with the variable name associated with the object.
    It can be useful for quickly inspecting the data structures.

    Args:
        *args: A list of either DataFrames or NumPy arrays, for which you want
               to display the shape.

    Examples:
        >>> my_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> my_array = np.array([[1, 2], [3, 4]])
        >>> zero_matrix = np.zeros((25, 6))

        >>> print_shapes(my_df, my_array, zero_matrix)
        my_df.shape = (3, 2)
        my_array.shape = (2, 2)
        zero_matrix.shape = (25, 6)
    """
    frame = inspect.currentframe()
    local_vars = frame.f_back.f_locals

    for arg in args:
        var_name = [name for name, value in local_vars.items() if value is arg][0]

        print(f"{var_name}.shape = {arg.shape}")
