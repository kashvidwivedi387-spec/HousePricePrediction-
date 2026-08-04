# 🏠 House Price Prediction System
### AIML Summer Internship 2026 — IIHMF, MNNIT Allahabad, Prayagraj

---

## Project Overview

A complete end-to-end Machine Learning solution that predicts house prices based on structural and location features, deployed as an interactive Streamlit web application.

## Folder Structure

```
HousePricePrediction/
│
├── Dataset/
│   └── house_prices.csv           # 2000-sample dataset
│
├── Notebook/
│   └── House_Price_Prediction.py  # Complete ML notebook (all 8 phases)
│
├── Model/
│   ├── best_model.pkl             # Trained Gradient Boosting model
│   ├── scaler.pkl                 # StandardScaler for Linear Regression
│   ├── label_encoder.pkl          # LabelEncoder for location feature
│   └── metrics.json               # Model performance metrics
│
├── Streamlit_App/
│   └── app.py                     # Interactive prediction web app
│
├── Documentation/
│   └── House_Price_Prediction_Report.pdf  # Full project report
│
└── README.md
```

## Machine Learning Models Used

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | ~$46,000 | ~$60,000 | 0.9046 |
| Random Forest | ~$6,500 | ~$9,300 | 0.9977 |
| **Gradient Boosting** ✓ | **~$5,700** | **~$7,600** | **0.9985** |

## Features Used

| Feature | Description |
|---|---|
| `area_sqft` | Total built-up area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Age of the property in years |
| `floors` | Number of floors |
| `garage_spaces` | Number of garage/parking spots |
| `has_pool` | Swimming pool (1=Yes, 0=No) |
| `has_garden` | Garden/yard (1=Yes, 0=No) |
| `location` | Neighbourhood category |
| `price_per_sqft` | Engineered: price per square foot estimate |
| `total_rooms` | Engineered: bedrooms + bathrooms |
| `amenity_score` | Engineered: pool + garden + garage |

## Setup & Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

### 2. Generate Dataset & Train Models
```bash
python generate_and_train.py
```

### 3. Run the Notebook
```bash
cd Notebook
python House_Price_Prediction.py
```

### 4. Launch the Web App
```bash
streamlit run Streamlit_App/app.py
```

## Submission Format

```
HousePricePrediction_<TeamLeaderName>.zip
```

---

*AIML Summer Internship 2026 | IIHMF, MNNIT Allahabad*
