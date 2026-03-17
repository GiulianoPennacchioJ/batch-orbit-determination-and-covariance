import pandas as pd
import numpy as np

def parse_gmd_file(filepath):
    """
    Parses a GMAT .gmd file (Range and RangeRate) for Orbit Determination.
    Skips header lines and comments starting with '%'.
    """
    observations = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            
            # Skip empty lines or comment lines starting with '%'
            if not parts or parts[0].startswith('%'):
                continue
            
            try:
                # 1. Time in A1 Modified Julian Date (MJD)
                time_mjd = float(parts[0])
                
                # 2. Type of measurement (Range or RangeRate)
                obs_type = parts[1]
                
                # 3. Value (always the last element in the line)
                value = float(parts[-1])
                
                # 4. Station ID Identification
                station = "Unknown"
                if "Santiago" in line: station = "Santiago"
                elif "Dongara" in line: station = "Dongara"
                
                observations.append({
                    'Time_MJD': time_mjd,
                    'Type': obs_type,
                    'Station': station,
                    'Value': value
                })
            except ValueError:
                # If a line still cannot be parsed as a float, skip it (extra safety)
                continue
            
    df = pd.DataFrame(observations)
    if df.empty:
        print(f"Warning: No valid data found in {filepath}. Check the file content.")
    else:
        print(f"Loaded {len(df)} measurements from GMD file.")
    return df

def load_ground_truth(filepath):
    """
    Loads the ground truth file from GMAT.
    Automatically detects if the separator is a space, comma, or tab.
    Cleans column names to extract X, Y, Z, VX, VY, VZ.
    """
    # Use sep=None and engine='python' to tell pandas to guess the separator
    df = pd.read_csv(filepath, sep=None, engine='python', skipinitialspace=True, comment='%')
    
    # Clean column names logic:
    new_columns = []
    for col in df.columns:
        # Remove quotes and extra spaces
        clean_name = col.strip().replace('"', '')
        # If GMAT wrote 'Sat.X', we take only 'X'
        if '.' in clean_name:
            clean_name = clean_name.split('.')[-1]
        new_columns.append(clean_name)
    
    df.columns = new_columns
    
    # Check if we actually found the required columns
    required = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    found_required = all(col in df.columns for col in required)
    
    print(f"Loaded {len(df)} truth states for validation.")
    print(f"Detected columns: {list(df.columns)}")
    
    if not found_required:
        print("ERROR: Could not find all required state columns (X, Y, Z, VX, VY, VZ).")
        print("Check if your GMAT ReportFile includes these parameters.")

    return df

def get_synchronized_truth(truth_df, first_obs_mjd):
    """
    Finds the truth state in the CSV that matches the first observation epoch.
    
    Inputs:
        truth_df: DataFrame from load_ground_truth
        first_obs_mjd: The MJD of the first observation in the .gmd file
        
    Returns:
        x_true: 6x1 numpy array [x, y, z, vx, vy, vz]
        matched_mjd: The exact MJD found in the truth file
    """
    # Find the row where the A1ModJulian is closest to our first observation
    # We use 'A1ModJulian' (or 'ModJulian' depending on your column cleaning)
    time_col = 'A1ModJulian' if 'A1ModJulian' in truth_df.columns else 'ModJulian'
    
    # Calculate absolute difference between each truth row and the target MJD
    diff = np.abs(truth_df[time_col] - first_obs_mjd)
    idx_match = diff.idxmin()
    
    matched_row = truth_df.iloc[idx_match]
    matched_mjd = matched_row[time_col]
    
    x_true = np.array([
        matched_row['X'], matched_row['Y'], matched_row['Z'],
        matched_row['VX'], matched_row['VY'], matched_row['VZ']
    ])
    
    print(f"Sincronizzazione completata!")
    print(f"  Target MJD:  {first_obs_mjd:.10f}")
    print(f"  Matched MJD: {matched_mjd:.10f} (Diff: {diff.min()*86400:.4f} sec)")
    
    return x_true, matched_mjd