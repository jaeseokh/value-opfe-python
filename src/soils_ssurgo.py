# ============================================
# Script: data_analysis.py
# Purpose: Train XGBoost, Random Forest, and PyTorch MLP models using leave-one-ffy-out cross-validation (parallelized)
# Outputs: MSE comparison for each method across field-years
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import time
from joblib import Parallel, delayed
import os


# ------------------------------
# Step 1: Load and prepare data
# ------------------------------
project_root = Path(".")
data_path = project_root / "Data" / "Processed" / "Analysis_ready"

# Load datasets
dat_binded = pd.read_csv(data_path / "dat_binded.csv")
info_binded = pd.read_csv(data_path / "info_binded.csv")
n_table = pd.read_csv(data_path / "n_table_anony.csv")

# Filter out fields with non-zero base N rate
base_ffy = n_table[n_table["N_base"] != 0]["ffy_id"].unique().tolist()

# Filter and subset usable columns
reg_columns = ["yield", "n_rate", "elev", "slope", "aspect", "prcp_t", "gdd_t", "ffy_id"]
dat_nozero = dat_binded[dat_binded["ffy_id"].isin(base_ffy)].copy()
dat_nozero = dat_nozero[reg_columns].dropna()

# ------------------------------
# Step 2: Define Model Classes and Utils
# ------------------------------

# MLP model: approximates function f(x) using fully connected neural network
# MLP: \hat{y} = W_3 * ReLU(W_2 * ReLU(W_1 * x + b_1) + b_2) + b_3
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Function to split data into features and labels
# X = df.drop(y), y = df[yield]
def split_xy(df):
    X = df.drop(columns=["yield", "ffy_id"])
    y = df["yield"]
    return X, y

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Step 3: Process Each Field-Year
# ------------------------------

def process_ffy(ffy):
    print(f"Processing field-year: {ffy}")
    test_data = dat_nozero[dat_nozero["ffy_id"] == ffy].copy()
    train_data = dat_nozero[dat_nozero["ffy_id"] != ffy].copy()

    x_train, y_train = split_xy(train_data)
    x_test, y_test = split_xy(test_data)

    # --- XGBoost: f^{(m)}(x) = f^{(m-1)}(x) + \eta * T_m(x) ---
    xgb_reg = XGBRegressor(objective='reg:squarederror')
    param_dist_xgb = {
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    search_xgb = RandomizedSearchCV(xgb_reg, param_dist_xgb, n_iter=4, cv=3, scoring='neg_mean_squared_error', n_jobs=1, random_state=42)
    search_xgb.fit(x_train, y_train)
    xgb_preds = search_xgb.best_estimator_.predict(x_test)

    # --- Random Forest: \hat{f}_{RF}(x) = (1/M) * sum_{m=1}^M T_m(x) ---
    rf_reg = RandomForestRegressor(random_state=42)
    param_dist_rf = {
        'n_estimators': [100, 200],
        'max_depth': [8, 10],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt']
    }
    search_rf = RandomizedSearchCV(rf_reg, param_dist_rf, n_iter=4, cv=3, scoring='neg_mean_squared_error', n_jobs=1, random_state=42)
    search_rf.fit(x_train, y_train)
    rf_preds = search_rf.best_estimator_.predict(x_test)

    # --- MLP: Train using SGD to minimize \sum (y_i - \hat{y}_i)^2 ---
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)

    mlp_model = MLP(input_dim=x_train.shape[1]).to(device)
    criterion = nn.MSELoss()  # loss = (1/n) * sum (y_i - f(x_i))^2
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

    mlp_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = mlp_model(x_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    torch_preds = mlp_model(x_test_tensor).detach().cpu().numpy().flatten()

    # --- MSE for each model: MSE = (1/n) * sum (y_i - \hat{y}_i)^2 ---
    mse_xgb = mean_squared_error(y_test, xgb_preds)
    mse_rf = mean_squared_error(y_test, rf_preds)
    mse_mlp = mean_squared_error(y_test, torch_preds)

    return {
        'ffy_id': ffy,
        'mse_xgb': mse_xgb,
        'mse_rf': mse_rf,
        'mse_mlp': mse_mlp
    }

# ------------------------------
# Step 4: Run in Parallel
# ------------------------------
start_time = time.time()
results = Parallel(n_jobs=-1, backend="loky")(delayed(process_ffy)(ffy) for ffy in base_ffy)
end_time = time.time()


# Ensure the path exists
output_dir = os.path.join(os.getcwd(), "Data", "Processed")
os.makedirs(output_dir, exist_ok=True)

# Save Results

# Save results
output_path = os.path.join(output_dir, "model_mse_by_ffy.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"✅ Parallel execution complete in {round(end_time - start_time, 2)} seconds.")
print(f"📁 Results saved to: {output_path}")
