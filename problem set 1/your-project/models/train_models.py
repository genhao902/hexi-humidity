import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1) Load data
# -----------------------------
# Use absolute path to the data folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "hexi_monthly_clean.csv")

# Fallback check
if not os.path.exists(DATA_PATH):
    DATA_PATH = "hexi_monthly_clean.csv"

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded: {DATA_PATH}")

# -----------------------------
# 2) Features & target
# -----------------------------
features = ["temperature_mean", "pressure_mean", "wind_speed_mean", "dew_point_mean"]
target = "humidity_mean"

X = df[features]
y = df[target]

# Keep time order: first 80% train, last 20% test
n = len(df)
split = int(n * 0.8)
X_train_raw, X_test_raw = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Standardization (essential for SGD gradient descent)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# -----------------------------
# 3) Train & evaluate helper
# -----------------------------
def train_eval(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    return {
        "name": name,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "pred": pred
    }

# -----------------------------
# 4) Run Models
# -----------------------------
ols_res = train_eval("Linear Regression (OLS)", LinearRegression())
sgd_res = train_eval("SGD Regressor", SGDRegressor(max_iter=5000, random_state=42))

results = [ols_res, sgd_res]
res_df = pd.DataFrame([r["metrics"] for r in results])
res_df.insert(0, "Model", [r["name"] for r in results])

# Save Metrics to data folder
csv_path = os.path.join(DATA_DIR, "model_metrics_comparison.csv")
res_df.to_csv(csv_path, index=False)
print(f" Metrics saved to: {csv_path}")

# -----------------------------
# 5) Functional Comparison
# -----------------------------
print("\n" + "="*70)
print(" FUNCTIONAL COMPARISON: OLS VS SGD")
print("="*70)
print(f"{'Feature':<25} | {'OLS (Normal Equation)':<22} | {'SGD (Iterative)'}")
print("-" * 70)
print(f"{'Method':<25} | {'Mathematical Exact':<22} | {'Gradient Descent'}")
print(f"{'Scaling':<25} | {'Optional':<22} | {'Required'}")
print(f"{'Best for':<25} | {'Small Datasets':<22} | {'Huge Datasets'}")
print("="*70)

# -----------------------------
# 6) Visualization (Bar Chart)
# -----------------------------
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelweight"] = "bold"

metrics_to_plot = ["MAE", "RMSE", "R2"]
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Model Performance Comparison", fontsize=18, fontweight='bold')

colors = ['#3498db', '#e67e22']

for i, metric in enumerate(metrics_to_plot):
    axes[i].bar(res_df["Model"], res_df[metric], color=colors, alpha=0.8)
    axes[i].set_title(metric, fontsize=14, fontweight='bold')
    axes[i].tick_params(axis='both', labelsize=11)
    # Add labels on top of bars
    for bar in axes[i].patches:
        axes[i].annotate(f'{bar.get_height():.3f}', 
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
img_path = os.path.join(DATA_DIR, "model_comparison_bar.png")
plt.savefig(img_path, dpi=300)
plt.show()
print(f"📈 Bar chart saved to: {img_path}")
