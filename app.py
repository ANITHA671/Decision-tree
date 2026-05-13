import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns

# Title
st.title("Decision Tree Classifier - Iris Dataset")

# Load Dataset
data = load_iris()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df['target'] = data.target

# Show Dataset
st.subheader("Dataset")
st.dataframe(df.head())

# Input and Output
X = df.drop('target', axis=1)
y = df['target']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Sidebar Parameters
st.sidebar.header("Model Parameters")

criterion = st.sidebar.selectbox(
    "Criterion",
    ['gini', 'entropy', 'log_loss']
)

max_depth = st.sidebar.slider(
    "Max Depth",
    1,
    10,
    3
)

splitter = st.sidebar.selectbox(
    "Splitter",
    ['best', 'random']
)

# Model
model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    splitter=splitter,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Accuracy Score")

st.write(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
)

st.text(report)

# Decision Tree Visualization
st.subheader("Decision Tree Visualization")

fig2, ax2 = plt.subplots(figsize=(12, 8))

plot_tree(
    model,
    filled=True,
    feature_names=data.feature_names,
    class_names=data.target_names,
    ax=ax2
)

st.pyplot(fig2)

# GridSearchCV
st.subheader("GridSearchCV")

params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [1, 2, 3, 4, 5, None],
    'splitter': ['best', 'random']
}

grid = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=params,
    cv=5
)

grid.fit(X_train, y_train)

st.write("Best Parameters:")
st.write(grid.best_params_)

st.write("Best Cross Validation Score:")
st.write(grid.best_score_)