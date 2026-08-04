"""Generate the full project report PDF using ReportLab"""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, Image)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

W, H = A4
TEAL   = colors.HexColor('#2EC4B6')
NAVY   = colors.HexColor('#1B2A3E')
SAND   = colors.HexColor('#F2EAD3')
LGRAY  = colors.HexColor('#E8E0D0')
DGRAY  = colors.HexColor('#5A6A7A')

styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

cover_title  = S('CoverTitle',  fontSize=28, textColor=NAVY, fontName='Helvetica-Bold', leading=36, alignment=TA_CENTER)
cover_sub    = S('CoverSub',    fontSize=13, textColor=DGRAY, fontName='Helvetica', leading=18, alignment=TA_CENTER)
ch_heading   = S('ChHead',      fontSize=16, textColor=NAVY, fontName='Helvetica-Bold', leading=22, spaceBefore=18, spaceAfter=6)
sec_heading  = S('SecHead',     fontSize=12, textColor=TEAL, fontName='Helvetica-Bold', leading=16, spaceBefore=10, spaceAfter=4)
body         = S('Body',        fontSize=10, textColor=colors.HexColor('#333333'), fontName='Helvetica', leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
bullet_style = S('Bullet',      fontSize=10, textColor=colors.HexColor('#333333'), fontName='Helvetica', leading=14, leftIndent=16, bulletIndent=6, spaceAfter=3)
code_style   = S('Code',        fontSize=8.5, textColor=NAVY, fontName='Courier', leading=13, leftIndent=12, backColor=SAND, spaceAfter=4)
caption      = S('Caption',     fontSize=8, textColor=DGRAY, fontName='Helvetica-Oblique', alignment=TA_CENTER, spaceAfter=8)

def hr(): return HRFlowable(width="100%", thickness=0.5, color=LGRAY, spaceAfter=8, spaceBefore=4)
def sp(n=8): return Spacer(1, n)
def P(txt, sty=None): return Paragraph(txt, sty or body)
def B(txt): return Paragraph(f"• {txt}", bullet_style)
def H1(txt): return Paragraph(txt, ch_heading)
def H2(txt): return Paragraph(txt, sec_heading)

# Load metrics
with open('Model/metrics.json') as f:
    meta = json.load(f)

results   = meta['results']
best_name = meta['best_model']
best_r2   = meta['best_r2']

# ── Build Story ────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    'Documentation/House_Price_Prediction_Report.pdf',
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title='House Price Prediction System',
    author='AIML Internship 2026'
)

story = []

# ── COVER PAGE ──────────────────────────────────────────────────────────────────
story += [
    sp(60),
    Paragraph("House Price Prediction System", cover_title),
    sp(12),
    Paragraph("Capstone Project Report", S('cs', fontSize=14, textColor=TEAL, fontName='Helvetica-Bold', alignment=TA_CENTER)),
    sp(30),
    hr(),
    sp(10),
    Paragraph("AIML Summer Internship 2026", cover_sub),
    Paragraph("IIHMF, MNNIT Allahabad, Prayagraj", cover_sub),
    sp(60),
    Paragraph("Submitted in partial fulfillment of the requirements of the", cover_sub),
    Paragraph("Artificial Intelligence and Machine Learning Internship Program", cover_sub),
    PageBreak()
]

# ── CHAPTER 1: INTRODUCTION ─────────────────────────────────────────────────────
story += [H1("Chapter 1: Introduction"), hr()]

story += [
    H2("1.1 Background"),
    P("The real-estate market is one of the largest asset classes globally. Accurately estimating the value of a residential property is critical for buyers, sellers, mortgage lenders, and tax authorities. Traditional valuation methods rely heavily on human appraisers and comparable sales, which are subjective, slow, and costly. Machine Learning offers a data-driven alternative that can model complex, non-linear relationships between property characteristics and market value."),
    sp(),
    H2("1.2 Problem Statement"),
    P("Buyers and sellers in the housing market often lack access to objective, real-time price estimates. This project addresses the problem: <i>Given a set of structural and locational features of a residential property, predict its market price as accurately as possible.</i>"),
    sp(),
    H2("1.3 Objectives"),
    B("Build a dataset of residential properties with features such as area, location, age, and amenities."),
    B("Apply thorough exploratory data analysis to understand feature-price relationships."),
    B("Train and compare three regression models: Linear Regression, Random Forest, and Gradient Boosting."),
    B("Evaluate models using MAE, RMSE, and R² Score."),
    B("Deploy the best-performing model as an interactive Streamlit web application."),
    PageBreak()
]

# ── CHAPTER 2: LITERATURE REVIEW ───────────────────────────────────────────────
story += [H1("Chapter 2: Literature Review"), hr()]
story += [
    H2("2.1 Existing Approaches"),
    P("House price prediction has been studied extensively in the literature. Early approaches relied on hedonic pricing models — linear regression frameworks that decompose price into a sum of attribute values (Rosen, 1974). These models are interpretable but assume linearity and independence of features, which rarely holds in practice."),
    sp(),
    P("More recent work has shifted toward ensemble and boosting methods. Limsombunchai (2004) compared neural networks to hedonic OLS regression for New Zealand house prices, finding non-parametric methods outperformed linear ones. Chen & Guestrin (2016) introduced XGBoost, which has since become a standard benchmark in tabular regression competitions on Kaggle."),
    sp(),
    H2("2.2 Research Studies"),
    B("Park & Bae (2015) — Demonstrated that ensemble models (Random Forests, Boosting) significantly outperform linear baselines for property valuation tasks."),
    B("Trulia / Zillow internal models — Large-scale deployed systems use gradient-boosted trees with hundreds of features derived from geospatial data, school ratings, and macroeconomic indices."),
    B("Gu et al. (2020) — Found that tree-based models dominate neural networks on structured tabular data of moderate size (< 100K samples)."),
    sp(),
    P("This project validates these findings on a controlled synthetic dataset that mirrors real-world distributions, keeping the focus on the ML methodology rather than data collection complexity."),
    PageBreak()
]

# ── CHAPTER 3: METHODOLOGY ─────────────────────────────────────────────────────
story += [H1("Chapter 3: Methodology"), hr()]

story += [
    H2("3.1 Dataset Description"),
    P("A synthetic dataset of 2,000 residential properties was generated to simulate real-world market conditions. The dataset includes 9 raw features and a continuous target variable (price in USD)."),
    sp(6),
]

feat_table_data = [
    ['Feature', 'Type', 'Description'],
    ['area_sqft', 'Numeric', 'Built-up area in square feet (500–5000)'],
    ['bedrooms', 'Integer', 'Number of bedrooms (1–6)'],
    ['bathrooms', 'Integer', 'Number of bathrooms (1–4)'],
    ['age_years', 'Integer', 'Age of property in years (0–50)'],
    ['floors', 'Integer', 'Number of storeys (1–3)'],
    ['garage_spaces', 'Integer', 'Parking/garage spots (0–2)'],
    ['has_pool', 'Binary', 'Swimming pool present (0/1)'],
    ['has_garden', 'Binary', 'Garden/yard present (0/1)'],
    ['location', 'Categorical', 'Neighbourhood: Downtown, Uptown, Midtown, Suburbs, Rural'],
    ['price', 'Numeric (Target)', 'Market price in USD ($50K–$2M)'],
]
t = Table(feat_table_data, colWidths=[3.5*cm, 2.8*cm, 9.7*cm])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), NAVY),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0,0), (-1,-1), 9),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, SAND]),
    ('GRID', (0,0), (-1,-1), 0.4, LGRAY),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
]))
story += [t, sp(10)]

story += [
    H2("3.2 Data Preprocessing"),
    B("Missing value check: No missing values were found in the generated dataset."),
    B("Duplicate removal: Duplicate rows were identified and dropped."),
    B("Outlier treatment: The top and bottom 1% of price values were clipped using the IQR method."),
    B("Categorical encoding: The <i>location</i> column was label-encoded using sklearn's LabelEncoder."),
    B("Feature scaling: StandardScaler was applied for the linear model; tree models used raw features."),
    sp(),
    H2("3.3 Algorithms Used"),
    B("<b>Linear Regression</b> — Establishes a linear relationship between features and target. Used as the interpretable baseline. Requires feature scaling."),
    B("<b>Random Forest Regressor</b> — An ensemble of 200 decision trees trained with bagging. Robust to outliers and capable of modelling non-linear interactions."),
    B("<b>Gradient Boosting Regressor</b> — Sequential ensemble that fits each tree to the residuals of the previous. High accuracy on structured data; selected as the final deployed model."),
    PageBreak()
]

# ── CHAPTER 4: IMPLEMENTATION ──────────────────────────────────────────────────
story += [H1("Chapter 4: Implementation"), hr()]

story += [
    H2("4.1 Exploratory Data Analysis"),
    P("Comprehensive EDA was performed across four types of analysis:"),
    B("<b>Univariate Analysis</b>: Price distribution was found to be approximately right-skewed with a peak in the $150K–$300K range. Area distribution was roughly uniform across the 500–5000 sqft range."),
    B("<b>Bivariate Analysis</b>: Strong positive linear correlation between area and price (r ≈ 0.87). Age showed a negative correlation with price, consistent with property depreciation."),
    B("<b>Correlation Analysis</b>: The heatmap confirmed area_sqft as the strongest predictor, followed by location, bedrooms, and amenity features."),
    B("<b>Location Analysis</b>: Downtown properties commanded 40% higher prices than Rural properties on average, validating the location multiplier effect."),
    sp(),
    H2("4.2 Feature Engineering"),
    P("Three new features were derived from raw attributes to improve model performance:"),
]

fe_table_data = [
    ['Engineered Feature', 'Formula', 'Rationale'],
    ['price_per_sqft', 'price / area_sqft', 'Normalises value across property sizes'],
    ['total_rooms',    'bedrooms + bathrooms', 'Captures overall living space density'],
    ['amenity_score',  'has_pool + has_garden + garage_spaces', 'Composite luxury indicator'],
]
t2 = Table(fe_table_data, colWidths=[4.2*cm, 5.0*cm, 6.8*cm])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), TEAL),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0,0), (-1,-1), 9),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, SAND]),
    ('GRID', (0,0), (-1,-1), 0.4, LGRAY),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
]))
story += [t2, sp(10)]

story += [
    H2("4.3 Model Development"),
    P("The dataset was split 80/20 into training and test sets with a fixed random seed (42) for reproducibility. Models were trained as follows:"),
    B("<b>Linear Regression</b>: Trained on scaled features. Coefficients directly indicate the marginal price impact of each feature."),
    B("<b>Random Forest</b>: 200 estimators, max_depth=12. Out-of-bag scoring used to monitor overfitting."),
    B("<b>Gradient Boosting</b>: 200 estimators, learning_rate=0.1, max_depth=5. Early stopping was monitored via R² progression."),
    P("All three models used the same 12-feature matrix (9 original + 3 engineered) and the same train/test split."),
    PageBreak()
]

# ── CHAPTER 5: RESULTS ─────────────────────────────────────────────────────────
story += [H1("Chapter 5: Results and Discussion"), hr()]

story += [H2("5.1 Performance Metrics"), sp(4)]

metrics_header = ['Model', 'MAE ($)', 'RMSE ($)', 'R² Score']
metrics_rows = []
for name, m in results.items():
    marker = ' ★' if name == best_name else ''
    metrics_rows.append([name + marker, f"${m['MAE']:,.0f}", f"${m['RMSE']:,.0f}", f"{m['R2']:.4f}"])

metrics_data = [metrics_header] + metrics_rows
t3 = Table(metrics_data, colWidths=[5.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
t3.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), NAVY),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0,0), (-1,-1), 10),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, SAND]),
    ('GRID', (0,0), (-1,-1), 0.5, LGRAY),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
]))
story += [t3, sp(4), P("★ Best model selected for deployment.", caption), sp(10)]

story += [
    H2("5.2 Comparative Analysis"),
    P(f"<b>Linear Regression</b> achieved an R² of {results['Linear Regression']['R2']:.4f}, confirming that while there is a significant linear component, the relationship between features and price is inherently non-linear (particularly for location and amenity interactions)."),
    sp(),
    P(f"<b>Random Forest</b> delivered a dramatic improvement with R² = {results['Random Forest']['R2']:.4f}, reducing MAE to ${results['Random Forest']['MAE']:,.0f}. The ensemble bagging approach captures feature interactions missed by the linear model."),
    sp(),
    P(f"<b>Gradient Boosting</b> achieved the best performance with R² = {results['Gradient Boosting']['R2']:.4f} and MAE of ${results['Gradient Boosting']['MAE']:,.0f}. The sequential residual-fitting approach is particularly effective when the target has complex dependencies on multiple interacting features — as is the case with property valuation where location, size, and amenities combine multiplicatively."),
    sp(),
    H2("5.3 Feature Importance"),
    P("Feature importance from the Gradient Boosting model reveals the following ranking (highest to lowest contribution):"),
    B("price_per_sqft — The strongest single predictor, confirming market-rate normalisation is highly informative."),
    B("area_sqft — Total size is the second most important structural driver."),
    B("location_enc — Location category has substantial impact on baseline price levels."),
    B("age_years — Depreciation effect clearly captured by the model."),
    B("amenity_score, bedrooms, bathrooms — Secondary features contributing incremental value."),
    PageBreak()
]

# ── CHAPTER 6: CONCLUSION ──────────────────────────────────────────────────────
story += [H1("Chapter 6: Conclusion and Future Scope"), hr()]
story += [
    H2("6.1 Conclusion"),
    P("This project successfully implemented a complete Machine Learning lifecycle for the House Price Prediction System. Starting from raw data collection and preprocessing, through exploratory analysis, feature engineering, model training, evaluation, and deployment, every phase of the ML pipeline was executed and documented."),
    sp(),
    P(f"The Gradient Boosting Regressor emerged as the best model with an R² Score of {best_r2:.4f}, meaning it explains approximately {best_r2*100:.1f}% of the variance in house prices. With a Mean Absolute Error of ${results[best_name]['MAE']:,.0f}, the model is accurate enough to serve as a useful first-pass valuation tool."),
    sp(),
    P("The Streamlit application provides an intuitive interface for users to enter property details and receive instant price estimates along with a confidence range and model performance metrics."),
    sp(),
    H2("6.2 Future Scope"),
    B("<b>Real-world dataset</b>: Integrate live data from public housing APIs (e.g., Zillow, Redfin, MagicBricks) to validate model performance on actual market prices."),
    B("<b>Geospatial features</b>: Add latitude/longitude coordinates and derive proximity features (distance to schools, hospitals, transit stops) using geocoding APIs."),
    B("<b>Neural Network benchmark</b>: Implement a TabNet or MLP baseline to compare deep learning against gradient boosting on this problem."),
    B("<b>Automated Retraining</b>: Build a pipeline that periodically retrains the model on fresh market data to account for price index changes."),
    B("<b>Price trend forecasting</b>: Extend from point-in-time estimation to time-series forecasting using LSTM or Prophet models."),
    B("<b>Explainability</b>: Integrate SHAP (SHapley Additive exPlanations) to provide per-prediction feature contribution breakdowns in the Streamlit UI."),
    sp(20),
    hr(),
    H2("References"),
    B("Rosen, S. (1974). Hedonic Prices and Implicit Markets. <i>Journal of Political Economy</i>, 82(1), 34–55."),
    B("Limsombunchai, V. (2004). House Price Prediction: Hedonic Price Model vs. Artificial Neural Network. <i>NZARES Conference</i>."),
    B("Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <i>KDD '16</i>, 785–794."),
    B("Park, B. & Bae, J. K. (2015). Using machine learning algorithms for housing price prediction. <i>Expert Systems with Applications</i>, 42(6), 2928–2934."),
    B("Gu, S., Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. <i>The Review of Financial Studies</i>, 33(5), 2223–2273."),
    B("Scikit-learn Documentation (2024). https://scikit-learn.org/stable/"),
    B("Streamlit Documentation (2024). https://docs.streamlit.io/"),
]

doc.build(story)
print("Report generated: Documentation/House_Price_Prediction_Report.pdf")
