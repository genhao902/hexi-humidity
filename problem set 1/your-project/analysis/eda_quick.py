# eda_quick.py
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
plt.rcParams["font.size"]=11
df = pd.read_csv("hexi_monthly_clean.csv")
print("Columns:", df.columns.tolist())
print (df.head(6))
print ("\nrow:", len(df), "Cities:",
       df['city'].nunique(),
       "Period:", df['year'].min(),  "-", 
       df['year'].max()     
       )

g= sns.relplot(data=df, x="month", y="humidity_mean",
             hue="city", kind="line", col="year", marker="o",
             facet_kws={'sharey': False,})

g.set_axis_labels("Month", "Mean Humidity (%)")
g.set_titles("Year {col_name}")
plt.tight_layout()
plt.savefig("fig_humidity_trends.png", dpi=180)


num=df[["temperature_mean", "pressure_mean", "wind_speed_mean", 
    "dew_point_mean", "humidity_mean"]]
corr=num.corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="vlag", fmt=".2f")
plt.title("fig_correlation_heatmap")
plt.tight_layout()
plt.savefig("fig_correlation_heatmap.png", dpi=180)

plt.show()

