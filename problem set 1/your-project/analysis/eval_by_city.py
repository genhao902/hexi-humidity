import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set visual style for high visibility
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelweight"] = "bold"

# Step 1: Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Step 2: Load data from the data folder
csv_input_path = os.path.join(DATA_DIR, "hexi_monthly_clean.csv")
df = pd.read_csv(csv_input_path)

df = df.sort_values(["year", "month", "city"]).reset_index(drop=True)

rows = []

for city, sub in df.groupby("city"):
    X = sub[["temperature_mean", "pressure_mean", "wind_speed_mean", "dew_point_mean"]]
    y = sub["humidity_mean"]
    
    # Time-ordered split: keep chronological order so the model never
    # "sees" future data during training (no random shuffle here).
    split_index = int(len(sub) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Fit the scaler ONLY on training data, then apply the same transform
    # to the test set. Fitting on all data would leak test statistics.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    models = {
        "Linear Regression": LinearRegression(),
        "SGD Regressor": SGDRegressor(max_iter=5000, random_state=42)
    }
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        # RMSE is in the same unit as humidity (%), easier to interpret than MSE.
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        rows.append({
            "city": city,
            "model": name,
            "MSE": mse,
            "RMSE": rmse,
            "r2": r2
        })

results_df = pd.DataFrame(rows).sort_values(["city", "RMSE"]).reset_index(drop=True)

# Step 3: Save results CSV to data
csv_output_path = os.path.join(DATA_DIR, "results_by_city.csv")
results_df.to_csv(csv_output_path, index=False)

# Step 4: Generate and save an Evaluation Figure
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="city", y="RMSE", hue="model")
plt.title("Model Performance Comparison (RMSE by City)", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("City", fontsize=14, fontweight='bold')
plt.ylabel("RMSE (Lower is better)", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Model", title_fontsize=13, fontsize=11, frameon=True, shadow=True)
plt.tight_layout()

eval_fig_path = os.path.join(DATA_DIR, "model_evaluation.png")
plt.savefig(eval_fig_path, dpi=180)
plt.show()

print("\n===== Evaluation Results =====")
print(results_df.to_string(index=False))
print(f"\n[INFO] CSV saved to: {csv_output_path}")
print(f"[INFO] Performance Chart saved to: {eval_fig_path}")




