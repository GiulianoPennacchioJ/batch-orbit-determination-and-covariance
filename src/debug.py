import numpy as np
import pandas as pd
import os
from data_loader import parse_gmd_file, load_ground_truth, get_synchronized_truth
import measurement_models
import navigation_utils

def debug_check():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    obs_path = os.path.join(project_root, 'data', 'satellite_observations.gmd')
    truth_path = os.path.join(project_root, 'data', 'ground_truth.csv')

    # 1. Carica i dati
    obs_df = parse_gmd_file(obs_path)
    truth_df = load_ground_truth(truth_path)
    
    # 2. Sincronizzazione Temporale (Logica Senior)
    a1_ref = truth_df['A1ModJulian'].iloc[0]
    utc_ref = truth_df['UTCModJulian'].iloc[0]
    time_offset = a1_ref - utc_ref
    
    obs0 = obs_df.iloc[0]
    t_obs0 = obs0['Time_MJD'] # Tempo in A1
    val_gmat = obs0['Value']
    
    # 3. Trova lo stato "True" di GMAT al tempo esatto della misura
    x_true_t0, matched_mjd = get_synchronized_truth(truth_df, t_obs0)
    
    # 4. Calcola cosa pensa Python
    coords = navigation_utils.get_station_coords()
    pos_st_eci, _ = measurement_models.get_station_eci(coords[obs0['Station']], t_obs0, time_offset)
    
    dist_python_1way = np.linalg.norm(x_true_t0[0:3] - pos_st_eci)
    
    print("\n" + "="*50)
    print(f" DEBUG 1a MISURA ({obs0['Station']})")
    print("="*50)
    print(f" GMAT File Value:   {val_gmat:.6f} km")
    print(f" Python Calc 1-Way: {dist_python_1way:.6f} km")
    print(f" Python Calc 2-Way: {dist_python_1way*2.0:.6f} km")
    print("-" * 50)
    print(f" Delta 1-Way: {val_gmat - dist_python_1way:.6f} km")
    print(f" Delta 2-Way: {val_gmat - dist_python_1way*2.0:.6f} km")
    print("="*50)

if __name__ == "__main__":
    debug_check()