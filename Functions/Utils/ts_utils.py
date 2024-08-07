import numpy as np

def window_it(sig, twin, tovp):
    """
    Window a time series into a matrix with a specific window size and overlap percentage.
    
    Parameters:
    - sig: 1D array-like, the time series data.
    - twin: int, the size of each window.
    - tovp: float, the number of samples for overlap between windows (0 to 1).
    
    Returns:
    - A 2D NumPy array where each row is a window of the time series.
    """
    step_size = int(twin - tovp)
    num_windows = (len(sig) - twin) // step_size + 1
    
    windows = np.array([sig[i * step_size : i * step_size + twin] for i in range(num_windows)])
    
    return windows