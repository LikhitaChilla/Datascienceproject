from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Load your data
df = pd.read_csv("test.csv")

# Encode target
df['satisfaction'] = df['satisfaction'].replace({'neutral or dissatisfied': 'Neutral/Dissatisfied'})
df['satisfaction'] = df['satisfaction'].map({'Neutral/Dissatisfied': 0, 'satisfied': 1})
X = df.drop(columns=["Unnamed: 0", "id", "Gate location", "satisfaction"],axis=1)
y = df['satisfaction']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include='number').columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Full pipeline with classifier
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Fit the pipeline
pipeline.fit(X, y)

# Save the pipeline
with open("model_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model pipeline trained and saved successfully as 'model_pipeline.pkl'")
