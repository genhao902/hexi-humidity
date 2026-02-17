# eda_quick.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths relative to this script
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)

# All outputs will be saved in the 'data' folder for convenience
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load data from the data folder
csv_path = os.path.join(DATA_DIR, "hexi_monthly_clean.csv")
df = pd.read_csv(csv_path)

# Set visual style for high visibility
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelweight"] = "bold"

print("Columns:", df.columns.tolist())
print(df.head(6))
print("\nrow:", len(df), "Cities:",
      df['city'].nunique(),
      "Period:", df['year'].min(), "-", 
      df['year'].max()     
      )

# Line plot for humidity trends
g = sns.relplot(data=df, x="month", y="humidity_mean",
               hue="city", kind="line", col="year", marker="o",
               facet_kws={'sharey': False}, linewidth=3)

g.set_axis_labels("Month", "Mean Humidity (%)", fontsize=14, fontweight='bold')
g.set_titles("Year {col_name}", fontsize=15, fontweight='bold')
plt.tight_layout()

# Save figure to the data folder
humidity_fig_path = os.path.join(DATA_DIR, "humidity_trends.png")
plt.savefig(humidity_fig_path, dpi=180)

# Correlation heatmap
num = df[["temperature_mean", "pressure_mean", "wind_speed_mean", 
         "dew_point_mean", "humidity_mean"]]
corr = num.corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="vlag", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()

# Save heatmap to the data folder
heatmap_fig_path = os.path.join(DATA_DIR, "heatmap.png")
plt.savefig(heatmap_fig_path, dpi=180)

print(f"\n[SUCCESS] Figures saved to: {DATA_DIR}")
plt.show() # Updated to allow display


