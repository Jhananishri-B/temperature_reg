import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Temperature Prediction", layout="wide")
st.title("🌡️ Temperature Prediction with XGBoost")

df = pd.read_csv("cleaned_data.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
st.write("Dataset Preview:", df.head())
target_col = "LandAverageTemperature"
X = df.drop(columns=[target_col])
y = df[target_col]
test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Performance Metrics")
st.write({
    "R2 Score": r2,
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae
})
st.subheader("📈 Actual vs Predicted")
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
st.pyplot(plt)
st.subheader("🖊️ Make Your Own Prediction")
input_dict = {}
for col in X.columns:
    val = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
    input_dict[col] = val

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
st.success(f"Predicted {target_col}: {prediction[0]:.2f}")
