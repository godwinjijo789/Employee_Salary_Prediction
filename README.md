# Employee Salary Prediction & Hidden Pattern

A Streamlit analytics dashboard for exploring employee salary data, comparing regression models, and discovering workforce patterns. The dashboard preserves the existing preprocessing and model pipeline while providing a responsive light/dark interface and interactive Plotly visualizations.

## Dashboard sections
- A branded **Emp Analysis** masthead with horizontal, icon-based top navigation
- **Dashboard:** view employee and salary KPIs, department filters, salary charts, and a 3D salary landscape using the configured employee CSV
- **Data Explorer:** department, gender, education, age, experience, and salary filters; salary distributions; workforce charts; correlation heatmap; and 3D exploration
- **Preprocessing:** actual missing-value, duplicate, and IQR outlier checks; fitted encoding details; training/test row counts
- **Statistical Analysis:** Pearson/Spearman correlations, confidence intervals, Welch two-group salary tests, and Kruskal-Wallis comparisons for three or more groups
- **Salary Predictor:** Linear Regression, Decision Tree, or Random Forest using the fitted preprocessing pipeline and original feature order
- **Pattern:** K-Means (fixed at K=2), cluster counts and profiles, silhouette score, salary distributions, and 2D/3D views
- **Model Evaluation:** actual MAE, MSE, RMSE, R², actual-versus-predicted plot, and residual diagnostics
- **Smart Insights:** data-derived department, education, experience, salary, and cluster comparisons
- **About Project:** current-run metrics, workflow, and interpretation limits

The app reports results calculated from the configured employee CSV. It does not substitute sample values for missing data or label clusters as salary tiers without inspecting their measured profiles.

## Required columns
The employee CSV must contain these exact column names:
- `Age`
- `Experience`
- `Department`
- `Education`
- `Gender`
- `Salary`

`Salary` is the regression target. The other five columns are used as prediction features.
The app expects age, experience, and salary to be numeric. Currency symbols and text inside the salary column should be cleaned before use. Department, education, and gender should contain categorical labels. The existing pipeline uses pandas column types to distinguish numeric from categorical features; convert numeric-coded category columns to text before running the app if they should be one-hot encoded.

## Install and run (Windows 11 / VS Code)
1. Extract this ZIP folder.
2. Open the extracted folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment (recommended):
   ```powershell
   python -m venv .venv
   ```
5. Install packages into that environment:
   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
6. Start the dashboard:
   ```powershell
   .venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
   ```
7. Open **Dashboard**. The included `data\employee_salary_cleaned.csv` loads automatically and powers the dashboard, EDA, preparation, prediction, clustering, model evaluation, and insights sections.
8. To switch between light and dark appearance, open the top-right menu and choose **Light** or **Dark** under **Theme**.

## Data, models, and results
The project includes `data\employee_salary_cleaned.csv` but does not include serialized trained model files. The app loads the CSV from the project directory; change `DEFAULT_DATASET_PATH` in `app.py` if you move it. Regression models and K-Means are fitted in the Streamlit session using the CSV and current settings. Metrics and insights displayed by the dashboard are computed from that run.

The regression preprocessing is fitted within each training pipeline: numeric features are median-imputed and standardized; features loaded as categorical are most-frequent-imputed and one-hot encoded. Rows without a numeric salary target are omitted from regression training. Missing feature values are handled by the fitted pipeline. Duplicate and IQR-flagged records are shown for review but are not automatically removed.

### Windows Application Control
If Windows reports that an application-control policy blocked a pandas or scikit-learn DLL, do not disable or bypass that policy. Use an organization-approved Python environment or ask the administrator to approve/provide the required runtime and packages. Reinstalling packages alone may not resolve a policy block.

## Files
- `app.py` — dashboard, existing preprocessing, model training, prediction, and clustering workflow
- `.streamlit/config.toml` — light/dark theme configuration
- `data/employee_salary_cleaned.csv` — default employee dataset used throughout the app
- `requirements.txt` — Python dependencies
- `README.md` — setup and project notes
