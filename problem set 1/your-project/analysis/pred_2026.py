import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import os

# Set visual style for high visibility
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelweight"] = "bold"

# 1. Paths Setup
# Use relative paths for better portability
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ANALYSIS_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "hexi_monthly_clean.csv")

def run_prediction_sgd():
    print(" Starting 2026 Humidity Prediction using SGD Model...")
    
    # 2. Load and Prepare Data
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f" Error: Data file not found at {CLEAN_DATA_PATH}")
        return

    df = pd.read_csv(CLEAN_DATA_PATH)
    features = ["temperature_mean", "pressure_mean", "wind_speed_mean", "dew_point_mean"]
    X = df[features]
    y = df["humidity_mean"]

    # 3. Scale Features and Train SGD Model
    # SGD is sensitive to feature scaling, so we must normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sgd_model = SGDRegressor(max_iter=5000, tol=1e-3, random_state=42)
    sgd_model.fit(X_scaled, y)
    print("SGD (Stochastic Gradient Descent) model trained successfully.")

    # 4. Generate 2026 Scenarios (Monthly averages per city)
    print("Generating 2026 weather scenarios...")
    cities = df["city"].unique()
    rows_2026 = []

    for city in cities:
        city_hist = df[df["city"] == city]
        for month in range(1, 13):
            month_stats = city_hist[city_hist["month"] == month][features].mean()
            row = {"city": city, "year": 2026, "month": month}
            for feat in features:
                row[feat] = month_stats[feat]
            rows_2026.append(row)

    df_2026 = pd.DataFrame(rows_2026)
    
    # 5. Execute Prediction
    # Important: Apply the SAME scaler used during training
    X_2026_scaled = scaler.transform(df_2026[features])
    df_2026["predicted_humidity"] = sgd_model.predict(X_2026_scaled)
    
    # Save Results
    results_path = os.path.join(DATA_DIR, "prediction_2026_sgd_results.csv")
    df_2026.to_csv(results_path, index=False)
    print(f"SGD Results saved to: {results_path}")

    # 6. Visualization: 2026 Trends
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_2026, x="month", y="predicted_humidity", hue="city", marker="o", linewidth=3)
    
    plt.title("2026 Predicted Air Humidity Trends (SGD Model)", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Month", fontsize=14, fontweight='bold')
    plt.ylabel("Predicted Humidity (%)", fontsize=14, fontweight='bold')
    plt.xticks(range(1, 13), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="City", title_fontsize=13, fontsize=11, frameon=True, shadow=True)
    
    trend_img = os.path.join(DATA_DIR, "prediction_2026_sgd_trends.png")
    plt.savefig(trend_img, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Trend visualization saved: {trend_img}")

    # 7. Model Insights
    print("\n--- SGD Model Insights ---")
    weights = pd.Series(sgd_model.coef_, index=features).sort_values(ascending=False)
    print("Feature Weights (Normalized):")
    for feat, weight in weights.items():
        print(f"🔹 {feat:<20}: {weight:.4f}")
    print("\nNote: Positive weight means the feature increases humidity; negative means it decreases it.")

if __name__ == "__main__":
    run_prediction_sgd()
