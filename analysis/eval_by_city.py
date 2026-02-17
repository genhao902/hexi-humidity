import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_squared_error, r2_score

df= pd.read_csv("hexi_monthly_clean.csv")

df= df.sort_values(["year", "month", "city"]).reset_index(drop=True)

rows=[]

for city, sub in df.groupby("city"):
    X= sub[["temperature_mean", "pressure_mean", "humidity_mean", "wind_speed_mean", "dew_point_mean"]]
    y= sub["humidity_mean"]
    
    
    split_index= int(len(sub)*0.8)
    X_train, X_test= X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test= y.iloc[:split_index], y.iloc[split_index:]
    
    models= {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred= m.predict(X_test)
        mse= mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2= r2_score(y_test, y_pred)
        rows.append({
            "city": city,
            "model": name,
            "MAE": mse,
            "RMSE": rmse,
            "r2": r2
        })
df = pd.DataFrame(rows).sort_values(["city","RMSE"]).reset_index(drop=True)
df.to_csv("Per-city Metrics.csv")
print("\n=====Per-city Metrics=====")
print(df.to_string(index=False))


