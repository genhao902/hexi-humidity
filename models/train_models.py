import pandas as pd, numpy as np
from  sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

df = pd.read_csv("hexi_monthly_clean.csv")

df.sort_values(["year", "month", "city"]).reset_index(drop=True)
X=df[["temperature_mean", "pressure_mean","wind_speed_mean", "dew_point_mean"]]
y = df["humidity_mean"]

n= len(df)
split= int(n*0.8)
X_train, X_test=X.iloc[: split], X.iloc[split:]
y_train, y_test=y.iloc[: split], y.iloc[split:]

def eval_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae= mean_absolute_error (y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2= r2_score (y_test, y_pred)
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}
    
results = []
results.append(eval_model("LinearRegression", LinearRegression()))
results.append(eval_model("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)))
res_df = pd.DataFrame(results).sort_values("RMSE")
res_df.to_csv("results_model_comparison.csv", index=False) 
print(res_df)
    
      


