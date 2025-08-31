



# TEMPERATURE Task      TEMPERATURE PREDICTOR APP LINK  -    https://temperaturereg-5bg7hfjjcau8tny9n5ftj7.streamlit.app/

This project analyzes global temperature data using Python, pandas, and scikit-learn. It includes data preprocessing, cleaning, and machine learning modeling for temperature prediction.

## Project Structure

- `app.py` : Main Python script (if used for running the app or analysis)
- `GlobalTemperatures.csv` : Raw dataset containing global temperature records
- `cleaned_data.csv` : Preprocessed and cleaned dataset
- `preprocessing.ipynb` : Jupyter notebook for data cleaning and feature engineering
- `CODE.ipynb` : Jupyter notebook for modeling, evaluation, and results visualization
- `visuals.ipynb` : Jupyter notebook for data visualization
- `requirements.txt` : Python dependencies

## Data Preprocessing

All preprocessing steps are performed in [`preprocessing.ipynb`](preprocessing.ipynb):

- Load data from `GlobalTemperatures.csv`
- Convert date column (`dt`) to datetime format
- Extract month as a new feature
- Drop the original date column
- Reset DataFrame index
- Remove rows with missing values
- Save the cleaned data to `cleaned_data.csv`

## Modeling

Modeling and evaluation are performed in [`CODE.ipynb`](CODE.ipynb):

- Load cleaned data
- Split data into training and testing sets
- Train regression models (e.g., Linear Regression, XGBoost)
- Evaluate models using metrics such as MSE, MAE, R², and Adjusted R²
- Display results in tables

## Visualization

Visualizations are created in [`visuals.ipynb`](visuals.ipynb):

- Plot temperature trends over time
- Visualize feature distributions and correlations

## Requirements

Install dependencies using:

```sh
pip install -r requirements.txt
```

## Usage

1. Run all cells in [`preprocessing.ipynb`](preprocessing.ipynb) to generate `cleaned_data.csv`.
2. Run [`CODE.ipynb`](CODE.ipynb) for modeling and evaluation.
3. Run [`visuals.ipynb`](visuals.ipynb) for data visualization.

## Notes

- Ensure `GlobalTemperatures.csv` is present in the workspace before running preprocessing.
- All notebooks use pandas and scikit-learn; XGBoost may be required for advanced modeling.

## License

This project is for educational
