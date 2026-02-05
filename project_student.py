# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1. Load data & Find missing values
df = pd.read_csv("Student_Perfor_predictor_test_PERFECT_1000.csv")  

print("Sample Rows")
print(df.head())

print("Dataset Shape")
print(f'Rows: {df.shape[0]}, Coluns: {df.shape[1]}')

print("Dataset info")
print(df.info())

print("Summary Statistics")
print(df.describe(include="all"))

print("Missing Values")
print(df.isnull().sum())


# Step2: Preprocessing(clean data)
print("Missing values in each column")
print(df.isnull().sum())

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender']) #like- M=0, F=1, O=2

print("After Encoding")
print(df.head())

print('Data-type after cleaning')
print(df.dtypes)

#Step3: FEATURE SCALING
features = [
    "study_hours","attendance_percentage","math_score",
    "science_score", "english_score"
]
# scaler = StandardScaler()
# df_scaled = df.copy()
# df_scaled[features] = scaler.fit_transform(df[features])

# TARGET COLUMNS (OUTPUT)
targets = ["math_score", "science_score", "english_score"]

X = df[features]
y = df[targets]

# Step4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step5: MODEL SLECTION
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Predict & evaluate
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R2: {r2:.3f}, MAE: {mae:.3f},MSE: {mse:.2f} ")

# 7. Save pipeline
joblib.dump(model, "student_model.joblib")


# joblib.dump(model_pipeline, "student_model_pipeline.joblib")
print("Saved model to student_model.joblib")

