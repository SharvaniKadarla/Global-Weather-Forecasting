# Weather Trend Forecasting

## PM Accelerator — AI Engineer Technical Assessment

A comprehensive data science and machine learning project for analyzing global weather patterns, forecasting future temperature trends, detecting anomalies, studying climate variability, analyzing weather–air-quality relationships, and identifying geographic patterns using the **Global Weather Repository** dataset.

---

## Project Overview

This project was developed as part of the **PM Accelerator AI Engineer Technical Assessment**.

The objective is to analyze historical global weather observations and build a complete data science workflow covering:

* Data cleaning and preprocessing
* Exploratory data analysis (EDA)
* Time-series feature engineering
* Weather trend forecasting
* Forecasting model comparison
* Ensemble modeling
* Feature importance analysis
* Multivariate anomaly detection
* Climate variability analysis
* Environmental and air-quality analysis
* Geographic and spatial analysis
* Data visualization
* Reproducible project organization

The project uses `last_updated` as the primary temporal feature and develops a forecasting case study using **Kyiv, Ukraine**.

> **Important scope note:** The forecasting model is demonstrated using Kyiv as a detailed case study rather than claiming that one location-specific model represents global weather behavior.

---

## PM Accelerator Mission

According to PM Accelerator's official AI Product Management page:

> "I'm on a mission to help launch 1,000+ AI products and empower professionals like you to become the next generation of AI product leaders — impacting millions of lives through real-world innovation."

Source: [PM Accelerator — AI Product Management](https://www.pmaccelerator.io/ai-product-management-certification)

This project applies data science, machine learning, analytical reasoning, and responsible interpretation to a real-world problem involving weather forecasting and environmental intelligence.

---

# Objectives

The project addresses the following technical objectives from the assessment:

### Basic Analysis

* Clean and preprocess the weather dataset
* Handle missing and invalid values
* Investigate outliers
* Perform exploratory data analysis
* Analyze temperature and precipitation patterns
* Identify correlations between weather variables
* Engineer time-series features
* Build a baseline forecasting approach
* Train and evaluate machine learning forecasting models

### Advanced Analysis

* Perform multivariate anomaly detection
* Compare multiple forecasting models
* Develop an ensemble forecasting model
* Analyze geographic climate variability
* Investigate relationships between weather and air quality
* Compare multiple feature-importance techniques
* Analyze geographic and spatial patterns
* Compare weather characteristics across countries and locations

---

# Dataset

The project uses the **Global Weather Repository** dataset available through Kaggle.

Dataset source:

https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

The dataset contains global weather observations together with geographic, temporal, and air-quality information.

### Dataset characteristics

After loading the data:

* **168,591 observations**
* **41 original columns**
* **211 countries**
* **268 raw location names**
* Coverage from **May 2024 through October 2026**
* `last_updated` used as the primary temporal field

The processed dataset contains additional engineered temporal features.

### Important temporal limitation

The available dataset spans approximately 2024–2026. Therefore, the climate analysis in this project is presented as **descriptive climate variability and geographic pattern analysis**, rather than as evidence of long-term climate change.

---

# Technologies Used

## Programming

* Python 3
* Jupyter Notebook

## Data Analysis

* pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Machine Learning

* scikit-learn
* Ridge Regression
* Random Forest Regression
* HistGradientBoosting Regression
* Isolation Forest
* StandardScaler
* Permutation Importance

## Model Persistence

* joblib
* JSON

## Development

* Visual Studio Code
* Git
* GitHub

---

# Project Structure

```text
Global-Weather-Forecasting/
│
├── data/
│   ├── raw/
│   │   └── Global Weather Repository.csv
│   │
│   └── processed/
│       └── weather_cleaned.csv
│
├── notebooks/
│   └── Weather_Trend_Forecasting_Analysis.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── coolest_countries.png
│   │   ├── co_by_humidity.png
│   │   ├── geographic_mean_pm10.png
│   │   ├── geographic_mean_pm25.png
│   │   ├── global_precipitation_spatial_distribution.png
│   │   ├── global_temperature_spatial_distribution.png
│   │   ├── kyiv_actual_vs_predicted.png
│   │   ├── no2_by_humidity.png
│   │   ├── ozone_by_humidity.png
│   │   ├── pm10_by_humidity.png
│   │   ├── pm2_5_by_humidity.png
│   │   ├── random_forest_feature_importance.png
│   │   ├── so2_by_humidity.png
│   │   ├── temperature_by_latitude_band.png
│   │   ├── temperature_vs_latitude.png
│   │   ├── top_anomaly_countries.png
│   │   ├── warmest_countries.png
│   │   └── weather_air_quality_correlation.png
│   │
│   ├── models/
│   │   ├── ridge_model.joblib
│   │   ├── random_forest_model.joblib
│   │   ├── hist_gradient_boosting_model.joblib
│   │   └── ensemble_weights.json
│   │
│   └── tables/
│       ├── anomaly_country_summary.csv
│       ├── climate_country_summary.csv
│       ├── environmental_impact_summary.csv
│       ├── feature_importance_comparison.csv
│       ├── forecasting_model_comparison.csv
│       ├── location_spatial_analysis.csv
│       └── pollution_hotspots.csv
│
├── report/
│   └── Weather_Trend_Forecasting_Report.pdf
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 1. Data Cleaning and Preprocessing

The original dataset was inspected for:

* Missing values
* Duplicate rows
* Invalid physical values
* Unit inconsistencies
* Suspicious temperature observations
* Unrealistic wind speeds
* Invalid atmospheric pressure values
* Negative air-quality measurements

### Initial quality checks

The original dataset contained:

* **0 missing values**
* **0 duplicate rows**

However, several physically implausible values were identified and investigated.

### Temperature validation

One extreme temperature record was identified:

* Location: Suva, Fiji Islands
* Temperature: approximately **79.3°C**
* The value was treated as invalid and replaced using location-level median imputation.

### Wind validation

Three suspicious wind observations exceeded physically reasonable thresholds.

These values were replaced and imputed using location-level medians.

### Pressure validation

Pressure values below 850 mb were not automatically removed because very low atmospheric pressure can occur in high-altitude locations.

Only clearly invalid extremely high pressure observations were corrected.

### Air-quality validation

Negative values were identified in several air-quality fields.

Invalid negative measurements were treated as missing and imputed using location-level medians.

### Extreme precipitation

A maximum precipitation value of approximately **74.52 mm** was retained because an extreme weather observation is not automatically an error.

### Unit consistency

Temperature and wind-unit conversions were also validated to ensure consistency between corresponding fields.

---

# 2. Final Data Quality

After preprocessing, the final dataset contained:

| Quality Check               |  Result |
| --------------------------- | ------: |
| Rows                        | 168,591 |
| Columns                     |      49 |
| Duplicate rows              |       0 |
| Missing values              |       0 |
| Invalid humidity values     |       0 |
| Invalid cloud values        |       0 |
| Negative precipitation      |       0 |
| Negative wind values        |       0 |
| Negative air-quality values |       0 |

The cleaned dataset is saved as:

```text
data/processed/weather_cleaned.csv
```

---

# 3. Temporal Feature Engineering

The `last_updated` column was converted into a datetime representation.

The following temporal features were created:

* `year`
* `month`
* `day`
* `day_of_year`
* `day_of_week`
* `week_of_year`
* `season`

Season categories were created as:

* Winter
* Spring
* Summer
* Fall

The data was then sorted chronologically by location.

---

# 4. Exploratory Data Analysis

The analysis examined distributions, seasonal patterns, correlations, and relationships between weather variables.

## Temperature

Key statistics:

* Mean: approximately **21.46°C**
* Median: **23.6°C**
* Standard deviation: approximately **9.29°C**
* Minimum: **-29.8°C**
* Maximum: **49.2°C**

## Precipitation

* Mean: approximately **0.132 mm**
* Median: **0 mm**
* Maximum: approximately **74.52 mm**

The large difference between the mean and median reflects the strongly right-skewed nature of precipitation observations.

---

# 5. Seasonal Temperature Analysis

Average temperature by season:

| Season | Mean Temperature |
| ------ | ---------------: |
| Winter |          16.60°C |
| Spring |          20.98°C |
| Summer |          24.88°C |
| Fall   |          21.60°C |

The monthly analysis showed the highest average temperatures during the middle of the year and lower averages during the winter months.

---

# 6. Seasonal Precipitation Analysis

Average precipitation by season:

| Season | Mean Precipitation |
| ------ | -----------------: |
| Winter |           0.113 mm |
| Spring |           0.124 mm |
| Summer |           0.142 mm |
| Fall   |           0.143 mm |

These values describe the distribution present in the dataset and should not be interpreted as globally representative climatological normals.

---

# 7. Correlation Analysis

Several notable relationships were identified.

Examples include:

* NO₂ and SO₂: approximately **0.69**
* PM2.5 and PM10: approximately **0.66**
* CO and PM2.5: approximately **0.61**
* CO and NO₂: approximately **0.61**
* Humidity and UV index: approximately **-0.53**
* Temperature and UV index: approximately **0.48**
* Temperature and pressure: approximately **-0.41**

The correlation analysis identifies statistical associations and does **not** establish causation.

Global pooled correlations can also be affected by geographic, seasonal, and regional differences.

---

# 8. Weather and Air-Quality Analysis

The project investigated relationships between weather variables and air-quality measurements including:

* Carbon monoxide (CO)
* Ozone
* Nitrogen dioxide (NO₂)
* Sulphur dioxide (SO₂)
* PM2.5
* PM10

Weather variables examined included:

* Temperature
* Precipitation
* Humidity
* Wind
* Pressure
* Visibility
* UV index

The analysis found several associations, including a negative relationship between humidity and several particulate-pollution measurements.

For example, the pooled correlation between humidity and:

* Ozone was approximately **-0.41**
* PM2.5 was approximately **-0.22**
* PM10 was approximately **-0.24**

Again, these results represent statistical associations rather than causal relationships.

---

# 9. Environmental Impact Analysis

Humidity-based groups were created:

* Low: `<40%`
* Moderate: `40–60%`
* High: `60–80%`
* Very High: `>80%`

Average particulate concentrations decreased across the humidity groups in the pooled dataset.

| Humidity Group | PM2.5 |   PM10 | Ozone |   NO₂ |   SO₂ |     CO |
| -------------- | ----: | -----: | ----: | ----: | ----: | -----: |
| Low            | 37.17 | 119.06 | 78.64 | 18.11 | 16.30 | 504.61 |
| Moderate       | 26.36 |  46.53 | 68.62 | 16.41 | 13.19 | 451.56 |
| High           | 19.84 |  31.05 | 55.11 | 12.51 |  8.12 | 413.29 |
| Very High      | 16.33 |  23.22 | 45.06 | 12.00 |  5.98 | 361.01 |

These results are descriptive associations in the dataset and should not be interpreted as evidence that humidity directly causes changes in pollutant concentrations.

---

# 10. Forecasting Case Study — Kyiv, Ukraine

A detailed forecasting case study was developed using **Kyiv, Ukraine**.

### Data coverage

* 867 observations
* 866 unique dates
* Approximately May 2024 – October 2026
* Daily aggregation used for modeling

Three missing calendar dates were identified in the Kyiv daily series and preserved as missing rather than artificially filling the target values.

The daily temperature series was used as the forecasting target.

---

# 11. Forecasting Feature Engineering

The forecasting dataset included:

### Calendar features

* Month
* Day of year
* Day of week
* Sinusoidal day-of-year feature
* Cosine day-of-year feature

### Lag features

* `lag_1`
* `lag_2`
* `lag_7`
* `lag_14`

### Rolling features

* `rolling_7`
* `rolling_14`

These features capture short-term persistence and seasonal patterns in temperature.

---

# 12. Chronological Train / Validation / Test Split

A chronological split was used to avoid future information leaking into model training.

| Dataset    | Rows | Period                  |
| ---------- | ---: | ----------------------- |
| Training   |  589 | 2024-05-30 → 2026-01-16 |
| Validation |  126 | 2026-01-17 → 2026-05-27 |
| Test       |  127 | 2026-05-28 → 2026-10-01 |

No random train/test split was used because the task is time-series forecasting.

---

# 13. Forecasting Models

Four forecasting approaches were evaluated:

### Baseline

Previous-day temperature (naive forecast)

### Model 1 — Ridge Regression

A Ridge regression model was implemented using:

* StandardScaler
* Ridge Regression
* `alpha = 1.0`

### Model 2 — Random Forest

Configuration:

* 300 trees
* Maximum depth: 12
* Minimum samples split: 5
* Minimum samples leaf: 2
* Random state: 42

### Model 3 — HistGradientBoosting

Configuration:

* 300 iterations
* Learning rate: 0.05
* Maximum leaf nodes: 15
* L2 regularization: 1.0
* Random state: 42

### Model 4 — Ensemble

An ensemble was created by combining:

* Ridge
* Random Forest
* HistGradientBoosting

Model weights were determined using inverse validation MAE.

---

# 14. Forecasting Results

## Validation Performance

| Model                | MAE (°C) | RMSE (°C) |    R² |
| -------------------- | -------: | --------: | ----: |
| Ridge                |    2.291 |     3.099 | 0.893 |
| Random Forest        |    2.989 |     4.281 | 0.795 |
| HistGradientBoosting |    2.794 |     3.781 | 0.840 |
| Ensemble             |    2.475 |     3.303 | 0.878 |

## Final Test Performance

| Model                | MAE (°C) | RMSE (°C) |    R² |
| -------------------- | -------: | --------: | ----: |
| Naive Baseline       |    2.695 |     4.078 | 0.234 |
| Ridge                |    2.355 |     3.527 | 0.427 |
| Random Forest        |    2.592 |     3.572 | 0.413 |
| HistGradientBoosting |    2.735 |     3.746 | 0.354 |
| Ensemble             |    2.424 |     3.505 | 0.434 |

The machine-learning approaches improved on the previous-day baseline on the final test period.

The Ridge model produced the lowest test MAE, while the ensemble produced the lowest test RMSE and highest test R² among the evaluated models.

The difference between validation and final test performance also highlights the importance of temporal distribution shifts and limited historical coverage.

The final test set was kept separate from model selection and was not used for further tuning.

---

# 15. Feature Importance

Feature importance was analyzed using two approaches:

1. Random Forest impurity-based importance
2. Permutation importance

The dominant predictive features were:

* `lag_1`
* `rolling_7`

Random Forest impurity importance:

| Feature         | Importance |
| --------------- | ---------: |
| lag_1           |     0.5651 |
| rolling_7       |     0.3831 |
| lag_2           |     0.0112 |
| cos_day_of_year |     0.0110 |
| rolling_14      |     0.0105 |

Permutation importance also ranked `lag_1` and `rolling_7` substantially above the remaining features.

This indicates that recent temperature history was the dominant predictive signal in the Kyiv case study.

Feature importance is interpreted as predictive association, not causation.

---

# 16. Anomaly Detection

A multivariate **Isolation Forest** was used to identify unusual weather and air-quality observations.

Features included:

* Temperature
* Precipitation
* Humidity
* Wind
* Pressure
* Visibility
* UV index
* PM2.5
* PM10

Configuration:

* `n_estimators = 200`
* `contamination = 0.01`
* `random_state = 42`

The model identified:

* **166,905 normal observations**
* **1,686 anomalous observations**
* Approximately **1%** of the dataset flagged as anomalies

---

# 17. Anomaly Characteristics

Compared with normal observations, flagged observations had:

* Higher average temperature
* Higher wind speed
* Higher UV index
* Lower humidity
* Lower visibility
* Much higher average PM2.5
* Much higher average PM10

The largest differences were observed in particulate pollution.

The anomaly analysis is intended to identify unusual multivariate patterns rather than declare individual observations erroneous.

Extreme observations were therefore not automatically removed.

---

# 18. Country-Level Anomaly Analysis

The anomaly analysis also examined the geographic distribution of flagged observations.

Examples of countries with high anomaly counts included:

| Country      | Observations | Anomalies | Anomaly Rate |
| ------------ | -----------: | --------: | -----------: |
| Saudi Arabia |          865 |       430 |       49.71% |
| Kuwait       |          865 |       245 |       28.32% |
| India        |          864 |       146 |       16.90% |
| Sudan        |        1,727 |       125 |        7.24% |
| Chile        |          863 |        96 |       11.12% |
| Chad         |          867 |        78 |        9.00% |
| Qatar        |          864 |        65 |        7.52% |
| China        |          864 |        63 |        7.29% |
| Bahrain      |          866 |        59 |        6.81% |
| Burkina Faso |          866 |        51 |        5.89% |

These results describe the distribution of statistical anomalies in this dataset and are not claims that the corresponding countries inherently have abnormal weather.

---

# 19. Climate Variability Analysis

Country-level temperature statistics were calculated to examine geographic climate variability.

Only countries with at least **100 observations** were considered for the main country-level comparisons to reduce the influence of locations with extremely sparse observations.

Examples of countries with high observed average temperatures included:

* Qatar
* United Arab Emirates
* Cambodia
* Oman
* Kuwait
* Djibouti
* Bangladesh
* Saudi Arabia
* Thailand
* Malaysia

Examples of countries with lower observed average temperatures included:

* Iceland
* Mongolia
* Canada
* Norway
* Andorra
* United States
* Finland
* Russia
* Estonia
* Lithuania

These results represent temperature patterns within the available dataset.

### Climate analysis limitation

Because the dataset covers approximately 2024–2026, this analysis should not be interpreted as a long-term climate-change study.

It is instead a descriptive analysis of geographic and seasonal climate variability in the available observations.

---

# 20. Latitude-Based Geographic Analysis

Temperature patterns were also analyzed by latitude bands.

| Latitude Band | Mean Temperature |
| ------------- | ---------------: |
| -60 to -30    |          13.55°C |
| -30 to 0      |          23.52°C |
| 0 to 30       |          26.40°C |
| 30 to 60      |          16.29°C |
| 60 to 90      |           7.62°C |

The results show a broad geographic relationship between latitude and observed temperature in the dataset.

This pattern is descriptive and influenced by the locations and observation frequencies represented in the source data.

---

# 21. Spatial Analysis

The dataset includes latitude and longitude coordinates, allowing geographic patterns to be explored.

The project analyzed:

* Global temperature distribution
* Global precipitation distribution
* Mean PM2.5 by location
* Mean PM10 by location
* Temperature versus latitude

The dataset covered:

* Latitude: approximately **-41.3° to 65.3°**
* Longitude: approximately **-175.2° to 179.22°**
* 211 countries

Location-level aggregation was used to compare average environmental characteristics across geographic locations.

---

# 22. Pollution Hotspots

Location-level particulate pollution was investigated.

Examples of locations with high mean PM2.5 included:

* Riyadh
* Santiago
* Jakarta
* Beijing
* New Delhi
* Kuwait City
* Hanoi
* Nouakchott
* Manama
* Dhaka

Examples of locations with high mean PM10 included:

* Riyadh
* Kuwait City
* Nouakchott
* New Delhi
* Juba
* Ouagadougou
* Manama
* Doha
* N'Djamena
* Santiago

These comparisons are based on the observations represented in the dataset and should not be interpreted as comprehensive real-world air-quality rankings.

---

# 23. Investigation of Extreme PM10 Values

Riyadh showed particularly high PM10 values.

A detailed investigation found:

* 865 observations
* Mean PM10: approximately **1,130.88**
* Median PM10: approximately **739.08**
* Maximum PM10: approximately **6,037.29**

The highest values occurred across multiple dates rather than being concentrated in one isolated observation.

Because the pattern was repeated across multiple observations, these values were retained for analysis rather than automatically classified as data-entry errors.

---

# 24. Model and Analysis Outputs

## Forecasting

```text
outputs/models/
├── ridge_model.joblib
├── random_forest_model.joblib
├── hist_gradient_boosting_model.joblib
└── ensemble_weights.json
```

## Tables

```text
outputs/tables/
├── anomaly_country_summary.csv
├── climate_country_summary.csv
├── environmental_impact_summary.csv
├── feature_importance_comparison.csv
├── forecasting_model_comparison.csv
├── location_spatial_analysis.csv
└── pollution_hotspots.csv
```

## Figures

The `outputs/figures/` directory contains:

* Temperature and precipitation visualizations
* Correlation analysis
* Forecast actual-vs-predicted visualization
* Feature importance visualization
* Anomaly analysis
* Climate comparisons
* Environmental impact plots
* Geographic and spatial visualizations

---

# 25. Reproducibility

The project includes:

* A complete Jupyter notebook
* `requirements.txt`
* Processed dataset
* Saved trained models
* Saved ensemble weights
* Saved analysis tables
* Saved visualization outputs
* `.gitignore`
* Project documentation

The primary analysis can be followed through:

```text
notebooks/Weather_Trend_Forecasting_Analysis.ipynb
```

---

# 26. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Global-Weather-Forecasting
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Launch Jupyter:

```powershell
jupyter notebook
```

Then open:

```text
notebooks/Weather_Trend_Forecasting_Analysis.ipynb
```

---

# 27. Running the Project

The main workflow is contained in the Jupyter notebook.

Recommended execution order:

1. Load the raw dataset
2. Inspect data types and structure
3. Check missing values and duplicates
4. Validate and clean suspicious values
5. Create temporal features
6. Perform exploratory data analysis
7. Analyze weather and air-quality relationships
8. Build the Kyiv forecasting dataset
9. Engineer lag and rolling features
10. Perform chronological train/validation/test splitting
11. Train forecasting models
12. Evaluate forecasting performance
13. Build the ensemble
14. Analyze feature importance
15. Perform Isolation Forest anomaly detection
16. Perform climate variability analysis
17. Perform environmental impact analysis
18. Perform spatial analysis
19. Save tables, figures, and trained models

---

# 28. Assessment Deliverables

| Assessment Requirement                   | Project Implementation                            |
| ---------------------------------------- | ------------------------------------------------- |
| Data cleaning/preprocessing              | Completed                                         |
| Missing-value analysis                   | Completed                                         |
| Outlier/anomaly analysis                 | Completed                                         |
| Normalization/scaling                    | Selective feature scaling for models requiring it |
| Basic EDA                                | Completed                                         |
| Temperature visualization                | Completed                                         |
| Precipitation visualization              | Completed                                         |
| Time-series feature using `last_updated` | Completed                                         |
| Forecasting model                        | Completed                                         |
| Evaluation metrics                       | MAE, RMSE, R²                                     |
| Multiple forecasting models              | Ridge, Random Forest, HistGradientBoosting        |
| Ensemble model                           | Completed                                         |
| Advanced anomaly detection               | Isolation Forest                                  |
| Feature importance                       | Impurity + permutation importance                 |
| Climate analysis                         | Country and latitude-based analysis               |
| Environmental impact                     | Weather–air-quality analysis                      |
| Spatial analysis                         | Latitude/longitude-based analysis                 |
| Country/geographic comparison            | Completed                                         |
| Report                                   | Included                                          |
| README                                   | Included                                          |
| Requirements file                        | Included                                          |

---

# 29. Demo Video

A short 1–2 minute demonstration video will show:

* Project overview
* Notebook workflow
* Data preprocessing
* Exploratory analysis
* Forecasting models
* Model evaluation
* Advanced analyses
* Key outputs

**Demo Video:** `ADD YOUR DEMO VIDEO LINK HERE`

---

# 30. Project Report

The complete written report will be available at:

```text
report/Weather_Trend_Forecasting_Report.pdf
```

The report documents:

* Data preparation
* Exploratory analysis
* Forecasting methodology
* Model evaluation
* Ensemble modeling
* Feature importance
* Anomaly detection
* Climate variability
* Environmental analysis
* Spatial analysis
* Limitations
* Conclusions

---

# 31. Key Findings

### Weather patterns

The dataset shows clear seasonal and geographic variation in temperature and precipitation.

### Forecasting

Recent temperature history, especially the previous day's temperature and 7-day rolling temperature average, provided the strongest predictive signals in the Kyiv case study.

### Model comparison

The evaluated machine-learning models improved upon the previous-day baseline on the final test period.

### Ensemble modeling

The ensemble provided competitive performance and achieved the highest test R² and lowest test RMSE among the evaluated machine-learning approaches.

### Anomaly detection

Approximately 1% of observations were identified as multivariate anomalies using Isolation Forest.

### Air quality

Particulate pollution showed substantial variation across humidity groups and geographic locations.

### Geographic patterns

Temperature varied systematically across latitude bands, while particulate concentrations showed strong geographic heterogeneity.

---

# 32. Limitations

Several limitations should be considered when interpreting the results.

### Limited historical period

The dataset covers approximately 2024–2026, which is insufficient for making claims about long-term climate change.

### Location-specific forecasting

The forecasting model was developed as a Kyiv case study and should not be interpreted as a globally generalizable forecasting model.

### Geographic representation

Observation counts are not uniform across countries and locations.

### Country-label inconsistencies

The source dataset contains some inconsistent country/location naming conventions.

### Correlation versus causation

Weather–air-quality relationships are observational associations and do not establish causal effects.

### Extreme values

Some extreme weather and air-quality observations may represent legitimate environmental events. Therefore, extreme values were not automatically removed solely because they were large.

### Model generalization

The difference between validation and test performance indicates that temporal distribution shifts can affect forecasting performance.

---

# 33. Responsible Interpretation

This project emphasizes analytical interpretation rather than overclaiming.

In particular:

* Correlation does not imply causation.
* Statistical anomalies are not automatically data errors.
* A short historical dataset cannot establish long-term climate trends.
* Country-level averages do not represent every location within a country.
* A single-city forecasting case study does not represent global forecasting performance.
* Model performance should be interpreted within the specific test period and feature set used.

---

# 34. Conclusion

This project demonstrates a complete end-to-end weather analytics and forecasting workflow using a large global weather dataset.

The workflow combines:

* Data engineering
* Statistical analysis
* Exploratory data analysis
* Time-series feature engineering
* Machine learning
* Ensemble modeling
* Explainability through feature importance
* Multivariate anomaly detection
* Environmental analysis
* Climate variability analysis
* Geographic analysis
* Data visualization

The resulting project provides both predictive modeling and broader environmental intelligence while explicitly documenting methodological limitations and responsible interpretation.

---

# 35. Author

**Sharvani Kadarla**

MS Computer Science — Software Development
Pace University

GitHub: `https://github.com/SharvaniKadarla`

LinkedIn: `https://www.linkedin.com/in/sharvani-kadarla`

---

# 36. References

### Dataset

Global Weather Repository — Kaggle:

https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

### PM Accelerator

PM Accelerator — AI Product Management:

https://www.pmaccelerator.io/ai-product-management-certification

### Project Repository

GitHub:

`https://github.com/SharvaniKadarla/Global-Weather-Forecasting`

---

## PM Accelerator Technical Assessment Submission

This repository was prepared as part of the **PM Accelerator AI Engineer Technical Assessment** and includes the requested analysis, forecasting models, visualizations, documentation, requirements, and project artifacts.
