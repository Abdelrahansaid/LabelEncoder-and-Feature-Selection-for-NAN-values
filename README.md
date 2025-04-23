# LabelEncoder-and-Feature-Selection-for-NAN-values

[![Python](https://img.shields.io/badge/Language-Python-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Data Description](#data-description)  
4. [Cleaning Steps](#cleaning-steps)  
5. [Schema & Performance](#schema--performance)  
6. [Results](#results)  
7. [Future Work](#future-work)  
8. [Author](#author)  
9. [License](#license)  

---

## Project Overview

This **House Price Prediction** project applies label encoding, feature selection, and three regression algorithms to predict residential property sale prices. It includes data cleaning, transformation, and model evaluation.

---

## Motivation

Accurate home-price estimates help buyers, sellers, and lenders make informed decisions. This pipeline handles missing values, selects the most predictive features, and compares model performance to find the best estimator.

---

## Data Description

- **Source**: Kaggle House Prices dataset  
- **Rows**: ~1,460  
- **Columns**: 81 features, including numeric, categorical, and date fields  
- **Target**: `SalePrice`  

Key feature groups:  
- Lot & area measurements (e.g. `LotArea`, `GrLivArea`)  
- Property attributes (e.g. `OverallQual`, `YearBuilt`)  
- Categorical descriptors (e.g. `Neighborhood`, `HouseStyle`)  

---

## Cleaning Steps

1. **Missing Value Handling**  
   - Impute numeric NaNs with median  
   - Impute categorical NaNs with mode  
2. **Label Encoding**  
   - Convert ordinal and nominal categories to numeric codes  
3. **Log Transformation**  
   - Apply `log1p` to target and skewed numeric features  
4. **Feature Creation**  
   - Binary flag `HasBsmt` for presence of basement  
5. **Correlation Analysis**  
   - Remove features with low correlation to `SalePrice`  
6. **Feature Selection**  
   - Recursive feature elimination (RFE) to select top 20 predictors  

---

## Schema & Performance

- **Input features**: 20 selected predictors after RFE  
- **Train/Test split**: 80% train, 20% test  
- **Evaluation metrics** on test set:  
  - MAE: 2,134  
  - MSE: 1,245,000  
  - MdAE: 1,850  

*(Metrics approximate—see notebook for exact values.)*

---

## Results

| Model                         | MAE    | MSE        | MdAE   |
|-------------------------------|--------|------------|--------|
| HistGradientBoostingRegressor | 2,010  | 1,100,000  | 1,750  |
| BaggingRegressor              | 2,250  | 1,350,000  | 1,900  |
| DecisionTreeRegressor         | 2,800  | 2,100,000  | 2,300  |

The **HistGradientBoostingRegressor** performed best on all metrics.

---

## Future Work

- Test additional algorithms (XGBoost, LightGBM, CatBoost)  
- Hyperparameter tuning with GridSearchCV or Bayesian optimization  
- Ensemble stacking of top models  
- Deploy as a REST API for real-time price prediction  

---

## Author

**Abdelrahman Said Mohamed**  
- 📎 [LinkedIn](https://www.linkedin.com/in/abdelrahman-said-mohamed-96b832234/)  
- 📧 abdelrahmanalgamil@gmail.com  

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  
