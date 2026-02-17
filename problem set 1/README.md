# 🌦️ Hexi Region Humidity Modeling & Prediction

This project focuses on analyzing historical meteorological data from the Hexi region and predicting air humidity trends for the year 2026 using Machine Learning models. The workflow covers data cleaning, Exploratory Data Analysis (EDA), model training/comparison, and future trend forecasting.

## 📊 Project Workflow (Core Steps)

Follow these steps in order to complete the analysis and generate predictions:

### Step 1: Environment Setup
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Step 2: Data Preparation & Cleaning
Extract and clean monthly data for cities in the Hexi region from raw datasets:
```bash
python scripts/build_hexi_monthly.py
```
*   **Artifact**: `data/hexi_monthly_clean.csv` (Cleaned monthly dataset)

### Step 3: Exploratory Data Analysis (EDA)
Analyze historical humidity trends and the correlation between weather factors (Temp, Wind Speed, etc.) and humidity:
```bash
python analysis/eda_quick.py
```
*   **Visual Artifacts**:
    *   `data/humidity_trends.png` (Historical seasonal trends)
    *   `data/heatmap.png` (Feature correlation heatmap)

### Step 4: Model Training & Evaluation
Train and compare **SGD (Stochastic Gradient Descent)** and **Linear Regression** models:
```bash
python models/train_models.py
```
*   **Metric Analysis**: Compare MAE, RMSE, and R² to determine model fit accuracy.
*   **Visual Artifact**: `data/model_comparison_bar.png` (Performance comparison chart)

### Step 5: Evaluation by City (Model Validation)
Individually evaluate prediction accuracy for each city to ensure model reliability across different geographical areas:
```bash
python analysis/eval_by_city.py
```
*   **Visual Artifact**: `data/model_evaluation.png` (RMSE comparison by city)

### Step 6: 2026 Humidity Prediction
The final step uses the optimized SGD model to forecast humidity changes for the entire year of 2026:
```bash
python analysis/pred_2026.py
```
*   **Final Results**:
    *   `data/prediction_2026_sgd_trends.png` (2026 Predicted Trends - **High Visibility Version**)
    *   `data/prediction_2026_sgd_results.csv` (Detailed prediction values)


---

## 📂 Project Structure
- **`analysis/`**: Scripts for data exploration and city-specific model evaluation.
- **`models/`**: Core machine learning training and performance comparison logic.
- **`data/`**: Central repository for datasets, cleaned data, generated charts, and results.
- **`scripts/`**: Utility scripts for initial data construction and preprocessing.
