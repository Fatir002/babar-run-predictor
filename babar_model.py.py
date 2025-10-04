import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\FATIR FARAZ\Downloads\archive (3)\babar-t20i-stats.csv')
df = pd.DataFrame(data)

# Drop columns that are not useful for prediction
df.drop(columns=['Min', 'SR', 'Start Date'], inplace=True)

# Clean the 'Runs' column (remove '*' and convert to integer)
df['Runs'] = df['Runs'].str.replace('*', '', regex=False).astype(int)

# Optional: Drop rows with missing values (just in case)
df.dropna(inplace=True)

# Drop categorical columns for now (since you're not encoding them yet)
df.drop(columns=['Dismissal', 'Opposition', 'Ground'], inplace=True)

# Feature scaling (standardization)
scaler = StandardScaler()
df[['BF', '4s', '6s', 'Pos']] = scaler.fit_transform(df[['BF', '4s', '6s', 'Pos']])

# Split features and target
X = df.drop(columns=['Runs'])
y = df['Runs']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"\nMAE: {mae}")
print(f"MSE: {mse}")
print("MSE / MAE:", mse / mae)
print(f"RMSE: {rmse}")
print(f"R² score: {r2}")

# Show correlation between numerical columns
print("\nCorrelation Matrix:")
print(df.corr())


# Scatter plot of 'BF' vs 'Runs'
plt.scatter(df['BF'], df['Runs'])
plt.xlabel('BF')
plt.ylabel('Runs')
plt.title('BF vs Runs')
plt.show()

# Actual vs Predicted Runs
plt.scatter(comparison['Actual'], comparison['Predicted'])
plt.plot([min(comparison['Actual']), max(comparison['Actual'])], 
         [min(comparison['Actual']), max(comparison['Actual'])], color='red')
plt.xlabel('Actual Runs')
plt.ylabel('Predicted Runs')
plt.title('Actual vs Predicted Runs')
plt.show()

