# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from imblearn.over_sampling import SMOTE
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report
# from joblib import dump

# # Load your dataset
# data = pd.read_csv("sqli_dataset.csv")

# # Handle NaN values by replacing NaN with empty strings and ensure the query is treated as a string
# data["query"] = data["query"].fillna("").astype(str)

# # Ensure labels (1 for SQLi, 0 for normal queries) are numeric and valid
# data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(-1).astype(int)

# # Features (SQL queries) and Labels (1 for SQLi, 0 for normal queries)
# X = data["query"]  # Feature
# y = data["label"]  # Target label

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Create a TfidfVectorizer
# vectorizer = TfidfVectorizer()

# # Vectorize the training data
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

# # Initialize SMOTE
# smote = SMOTE(random_state=42)

# # Apply SMOTE to the vectorized training data
# X_resampled, y_resampled = smote.fit_resample(X_train_vectorized, y_train)

# # Create a pipeline with TfidfVectorizer and Naive Bayes Classifier
# pipeline = Pipeline(
#     [
#         # ("tfidf", TfidfVectorizer()),  # Step 1: Vectorization
#         ("nb", MultinomialNB()),  # Step 2: Naive Bayes classification
#     ]
# )

# # Train the pipeline on the resampled data
# pipeline.fit(X_resampled, y_resampled)

# # Evaluate the pipeline on the test set
# y_pred = pipeline.predict(X_test_vectorized)

# # Print the classification report
# print(classification_report(y_test, y_pred))

# # Save the trained pipeline
# dump(pipeline, "sqli_model.pkl")

# print("Pipeline model trained and saved as sqli_pipeline_model.pkl")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump
import re

# Load your dataset
data = pd.read_csv("sqli_dataset.csv")


# Function for regex-based SQLi detection
def check_regex_sql_patterns(query):
    regex_patterns = [
        r"(\%27)|(\')|(\-\-)|(\%23)|(#)",  # Detects single quotes, comments like -- or #
        r"(\b(SELECT|DROP|UNION|INSERT|UPDATE|DELETE)\b)",  # Common SQL keywords
        r"\bOR\s+1\s*=\s*1\b",  # Detects OR 1=1
        r"(\bUNION\b\s*(\bALL\b)?\s*\bSELECT\b)",  # Detects UNION SELECT attacks
        r"(\bWHERE\b.*\s*\=\s*\')",  # WHERE clauses with suspicious single quotes
    ]
    for pattern in regex_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return 1  # SQLi Detected
    return 0  # No SQLi detected


# # Function for basic string-based SQLi detection
# def check_basic_sql_patterns(query):
#     sql_keywords = [
#         "SELECT", "DROP", "INSERT", "UPDATE", "DELETE", "UNION", "--", "#", "/*", "*/", " OR ", "' OR",
#     ]
#     for keyword in sql_keywords:
#         if keyword in query.upper():
#             return 1  # Indicates possible SQLi
#     return 0  # Indicates normal query


# # Apply pattern function to detect SQLi in your dataset
# data["basic_sqli_flag"] = data["query"].apply(check_basic_sql_patterns)
data["regex_sqli_flag"] = data["query"].apply(check_regex_sql_patterns)

# Handle NaN values by replacing NaN with empty strings and ensure the query is treated as a string
data["query"] = data["query"].fillna("").astype(str)

# Ensure labels (1 for SQLi, 0 for normal queries) are numeric and valid
data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(-1).astype(int)

# Features (SQL queries) and Labels (1 for SQLi, 0 for normal queries)
X = data["query"]  # Feature
y = data["label"]  # Target label

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Create a pipeline with TfidfVectorizer and Naive Bayes Classifier
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),  # Step 1: Vectorization
        ("nb", MultinomialNB()),  # Step 2: Naive Bayes classification
    ]
)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test set
y_pred = pipeline.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Save the trained pipeline
dump(pipeline, "sqli_model.pkl")

print("Pipeline model trained and saved as sqli_model.pkl")