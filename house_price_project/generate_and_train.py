"""
House Price Prediction System
Generates synthetic dataset, trains models, saves artifacts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# ── Dataset Generation ─────────────────────────────────────────────────────────
N = 2000
locations = ['Downtown', 'Suburbs', 'Rural', 'Midtown', 'Uptown']
loc_price = {'Downtown': 1.4, 'Uptown': 1.2, 'Midtown': 1.0, 'Suburbs': 0.85, 'Rural': 0.6}

area      = np.random.randint(500, 5000, N)
bedrooms  = np.random.randint(1, 7, N)
bathrooms = np.random.randint(1, 5, N)
age       = np.random.randint(0, 50, N)
floors    = np.random.randint(1, 4, N)
garage    = np.random.randint(0, 3, N)
pool      = np.random.choice([0, 1], N, p=[0.7, 0.3])
garden    = np.random.choice([0, 1], N, p=[0.5, 0.5])
location  = np.random.choice(locations, N)

base_price = (
    area * 120
    + bedrooms * 8000
    + bathrooms * 6000
    - age * 1500
    + floors * 5000
    + garage * 7000
    + pool * 20000
    + garden * 10000
)
loc_mult  = np.array([loc_price[l] for l in location])
noise     = np.random.normal(0, 15000, N)
price     = (base_price * loc_mult + noise).clip(50000, 2000000).astype(int)

df = pd.DataFrame({
    'area_sqft': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
    'age_years': age, 'floors': floors, 'garage_spaces': garage,
    'has_pool': pool, 'has_garden': garden, 'location': location, 'price': price
})

os.makedirs('Dataset', exist_ok=True)
df.to_csv('Dataset/house_prices.csv', index=False)
print(f"Dataset saved: {df.shape}")

# ── EDA Visualizations ─────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('House Price EDA', fontsize=16, fontweight='bold')

axes[0,0].hist(df['price']/1e6, bins=40, color='#2EC4B6', edgecolor='white')
axes[0,0].set_title('Price Distribution'); axes[0,0].set_xlabel('Price (Millions $)')

axes[0,1].scatter(df['area_sqft'], df['price']/1e6, alpha=0.3, color='#1B2A3E', s=10)
axes[0,1].set_title('Area vs Price'); axes[0,1].set_xlabel('Area (sqft)'); axes[0,1].set_ylabel('Price (M$)')

avg_by_loc = df.groupby('location')['price'].mean().sort_values()
axes[0,2].barh(avg_by_loc.index, avg_by_loc.values/1e6, color='#2EC4B6')
axes[0,2].set_title('Avg Price by Location'); axes[0,2].set_xlabel('Avg Price (M$)')

axes[1,0].boxplot([df[df['bedrooms']==b]['price'].values/1e6 for b in range(1,7)],
                  labels=range(1,7), patch_artist=True,
                  boxprops=dict(facecolor='#2EC4B6', alpha=0.6))
axes[1,0].set_title('Price by Bedrooms'); axes[1,0].set_xlabel('Bedrooms'); axes[1,0].set_ylabel('Price (M$)')

num_cols = ['area_sqft','bedrooms','bathrooms','age_years','floors','garage_spaces','price']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[1,1], cmap='coolwarm', annot=True, fmt='.2f', square=True, cbar=False)
axes[1,1].set_title('Correlation Heatmap')

axes[1,2].scatter(df['age_years'], df['price']/1e6, alpha=0.3, color='#FF6B6B', s=10)
axes[1,2].set_title('Age vs Price'); axes[1,2].set_xlabel('Age (years)'); axes[1,2].set_ylabel('Price (M$)')

plt.tight_layout()
os.makedirs('Notebook', exist_ok=True)
plt.savefig('Notebook/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA plots saved")

# ── Preprocessing ──────────────────────────────────────────────────────────────
le = LabelEncoder()
df['location_enc'] = le.fit_transform(df['location'])
df['price_per_sqft'] = df['price'] / df['area_sqft']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['amenity_score'] = df['has_pool'] + df['has_garden'] + df['garage_spaces']

features = ['area_sqft','bedrooms','bathrooms','age_years','floors','garage_spaces',
            'has_pool','has_garden','location_enc','price_per_sqft','total_rooms','amenity_score']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Model Training ─────────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
}

results = {}
best_r2, best_name, best_model = -1, None, None

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_sc, y_train); preds = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train); preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, preds)
    results[name] = {'MAE': round(mae,2), 'MSE': round(mse,2), 'RMSE': round(rmse,2), 'R2': round(r2,4)}
    print(f"{name:25s}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}")

    if r2 > best_r2:
        best_r2, best_name, best_model = r2, name, model

print(f"\nBest model: {best_name} (R²={best_r2:.4f})")

# ── Save Artifacts ─────────────────────────────────────────────────────────────
os.makedirs('Model', exist_ok=True)
joblib.dump(best_model, 'Model/best_model.pkl')
joblib.dump(scaler,     'Model/scaler.pkl')
joblib.dump(le,         'Model/label_encoder.pkl')

with open('Model/metrics.json', 'w') as f:
    json.dump({'results': results, 'best_model': best_name, 'best_r2': best_r2,
               'features': features, 'locations': list(le.classes_)}, f, indent=2)

print("Models & artifacts saved to Model/")

# ── Model Comparison Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
names = list(results.keys())
r2s   = [results[n]['R2'] for n in names]
colors = ['#2EC4B6' if n == best_name else '#1B2A3E' for n in names]
bars = ax.bar(names, r2s, color=colors, width=0.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison — R² Score')
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f'{val:.4f}',
            ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('Notebook/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Model comparison plot saved")
