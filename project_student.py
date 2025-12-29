# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1. Load data
df = pd.read_csv("Student_Perfor_predictor_test_PERFECT_1000.csv")  


# 2. FEATURE COLUMNS (INPUT)
features = [
    "study_hours",
    "attendance_percentage",
]

# # 3. TARGET COLUMNS (OUTPUT)
targets = ["math_score", "science_score", "english_score"]

numeric_features = features
categorial_features = ["gender"]

X = df[features + categorial_features]
y = df[targets]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Preprocessing pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_features),
    ("cat", cat_pipeline, categorial_features),
])

# 6. Full pipeline with model
model_p = Pipeline([
    ("preproc", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=30))
])

# 7. Fit
model_p.fit(X_train, y_train)
model_p.fit(X, y)

# 8. Predict & evaluate
y_pred = model_p.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2: {r2:.3f}, MAE: {mae:.3f}")

# 9. Save pipeline
joblib.dump(model_p, "student_model.joblib")

# joblib.dump(model_pipeline, "student_model_pipeline.joblib")
print("Saved model to student_model.joblib")
