# model_utils.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def prepare_dataset(interaction_data):
    states, actions, rewards, next_states, dones = zip(*interaction_data)
    X = np.hstack([states, actions, rewards])
    y = dones
    return np.array(X), np.array(y)


def train_ml_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model
