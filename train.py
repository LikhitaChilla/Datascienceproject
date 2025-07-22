import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load and clean dataset
df = pd.read_csv("test.csv")

# Target column preprocessing
df['satisfaction'] = df['satisfaction'].replace({'neutral or dissatisfied': 'Neutral/Dissatisfied'})
df['satisfaction'] = df['satisfaction'].map({'Neutral/Dissatisfied': 0, 'satisfied': 1})

# Drop unwanted columns
df = df.drop(columns=["Unnamed: 0", "id", "Gate location"], errors='ignore')

# Separate features and target
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

# Identify feature types
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include='number').columns.tolist()

# Build pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=100))
])

# Train and save
pipeline.fit(X, y)

with open("model_pipelines.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as model_pipeline.pkl")
