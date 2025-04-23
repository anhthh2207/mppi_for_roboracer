import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def downsample_df(df, target_length):
    """
    Downsample a DataFrame to a specified target length by selecting equally spaced rows.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be downsampled.
    target_length (int): The desired number of rows in the downsampled DataFrame.

    Returns:
    pd.DataFrame: A new DataFrame with the specified number of rows, containing
                  equally spaced rows from the original DataFrame. If the target
                  length is greater than or equal to the original DataFrame length,
                  a copy of the original DataFrame is returned.
    """
    """"""
    n = len(df)
    if target_length >= n:
        return df.copy()  # No downsampling needed
    # Generate equally spaced indices
    indices = np.linspace(0, n - 1, target_length, dtype=int)
    return df.iloc[indices].reset_index(drop=True)

def compute_frenet_s(df, x_col, y_col):
    """
    Compute the cumulative arc length (s) along a path defined by x and y coordinates.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the path data.
    x_col (str): The name of the column in the DataFrame containing x-coordinates.
    y_col (str): The name of the column in the DataFrame containing y-coordinates.

    Returns:
    np.ndarray: A 1D array representing the cumulative arc length (s) at each point
                along the path, starting from 0 at the first point.
    """
    x  = np.array(df[x_col])
    y  = np.array(df[y_col])
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0)
    return s


def processed_df(raw_df):
    """
    Processes a raw DataFrame containing positional data and computes additional 
    fields for further analysis.
    Args:
        raw_df (pd.DataFrame): A DataFrame containing raw positional data with at 
            least the columns 'x_position' and 'y_position'.
    Returns:
        pd.DataFrame: A new DataFrame with the following columns:
            - 's_m': Frenet longitudinal distance (computed from 'x_position' and 'y_position').
            - 'x_m': X-coordinate position (copied from 'x_position').
            - 'y_m': Y-coordinate position (copied from 'y_position').
            - 'psi_rad': Placeholder for orientation in radians (initialized as NaN).
            - 'kappa_radpm': Placeholder for curvature in radians per meter (initialized as NaN).
            - 'vx_mps': Placeholder for velocity in meters per second (initialized as NaN).
            - 'ax_mps2': Placeholder for acceleration in meters per second squared (initialized as NaN).
            - 'w_tr_right_m': Placeholder for right wheel track width in meters (initialized as NaN).
            - 'w_tr_left_m': Placeholder for left wheel track width in meters (initialized as NaN).
    """
    """"""
    n = len(raw_df)
    s = compute_frenet_s(raw_df, 'x_position', 'y_position')

    new_df = pd.DataFrame({
        '# s_m': s,
        ' x_m': raw_df['x_position'].values,
        ' y_m': raw_df['y_position'].values,
        ' psi_rad': 0,
        ' kappa_radpm': 0,
        ' vx_mps': 0,
        ' ax_mps2': 0,
        ' w_tr_right_m': 0,
        ' w_tr_left_m': 0
    })

    return new_df

if __name__ == "__main__":
    raw_file_path = './slam_levine/slam_levine_raw.csv'
    raw_df = pd.read_csv(raw_file_path)

    downsampled = downsample_df(raw_df, 640)
    processed = processed_df(downsampled)

    # Save the downsampled data to a new CSV file
    output_file_path = './slam_levine/slam_levine.csv'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file_path, 'w') as f:
        f.write(f"# {current_time}\n# dummytext\n")
        processed.to_csv(f, index=False, sep=';')

    print(f"Downsampled data saved to {output_file_path}")