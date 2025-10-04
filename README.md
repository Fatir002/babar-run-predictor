# babar-run-predictor
# 🏏 Babar Azam T20I Run Predictor

This project uses **Linear Regression** to predict the number of runs Babar Azam might score
in a T20 International (T20I) match, based on match-related features such as balls faced,
4s, 6s, and batting position.

## 📁 Project Structure

babar-run-predictor/  
├── babar-t20i-stats.csv # Dataset with Babar Azam's T20I match data  
├── babar_model.py       # Python code to preprocess data, train, and evaluate the model  
├── README.md            # Project description and instructions (this file)  

## 📊 Dataset

The dataset contains Babar Azam’s T20I performance stats with columns like:

- `Runs` – Runs scored (target column)
- `BF` – Balls faced
- `4s`, `6s` – Number of boundaries hit
- `Pos` – Batting position
- `Dismissal`, `Opposition`, `Ground` – Removed to simplify modeling

> Note: `Runs` values with `*` (not out) are cleaned and converted to integers.

---

## ⚙️ Technologies Used

- Python  
- pandas  
- numpy  
- scikit-learn  
- metplotlib
---

## 🚀 How to Run the Project

```bash
# 🏏 Babar Azam T20I Run Predictor

# Step 1: Place the dataset in the same directory
# Ensure babar-t20i-stats.csv is in the same folder as your Python script.

# Step 2: Install required Python libraries
pip install pandas numpy scikit-learn metplotlib

# Step 3: Run the model script
python babar_model.py

# Example Output:
# MAE: 2.55
# MSE: 10.56
# RMSE: 3.25
# R² score: 0.98
# MSE / MAE: 4.137149719127922

# R² score: 0.98
# MSE / MAE: 4.137149719127922
