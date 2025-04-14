
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('../data/netflix_titles.csv')

# Clean duration to numeric
def extract_duration(val):
    if pd.isnull(val):
        return None
    if "min" in val:
        return int(val.replace(" min", ""))
    elif "Season" in val:
        return int(val.split()[0])
    return None

df["duration_int"] = df["duration"].apply(extract_duration)

# Prepare data
ml_df = df[['type', 'listed_in', 'release_year', 'duration_int']].dropna()

# Encode target
le = LabelEncoder()
ml_df['type_encoded'] = le.fit_transform(ml_df['type'])

# Vectorize genres
tfidf = TfidfVectorizer(max_features=20, stop_words='english')
genre_features = tfidf.fit_transform(ml_df['listed_in'])

# Combine features
X = np.hstack([genre_features.toarray(),
               ml_df[['release_year', 'duration_int']].values])
y = ml_df['type_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)
