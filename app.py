
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

# ---------------------- CUSTOM CSS ----------------------

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

/* Main Title */
h1 {
    text-align: center;
    color: #4f46e5;
    font-size: 45px !important;
    font-weight: bold;
    padding: 15px;
    border-radius: 15px;
    background: linear-gradient(to right, #c7d2fe, #e0e7ff);
    border: 3px solid #6366f1;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* Subheaders */
h2, h3 {
    color: #1e293b;
    border-left: 6px solid #6366f1;
    padding-left: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #4338ca, #6366f1);
}

/* Sidebar Text */
section[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: bold;
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    border: 3px solid #6366f1;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* Metric Styling */
[data-testid="metric-container"] {
    background-color: white;
    border: 3px solid #6366f1;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* Plot Container */
.css-1kyxreq {
    border: 3px solid #6366f1;
    border-radius: 15px;
    padding: 10px;
    background-color: white;
}

/* Text Box */
pre {
    border: 2px solid #6366f1 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    background-color: #f8fafc !important;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("🌸 Decision Tree Classifier - Iris Dataset")

st.markdown(
    "<h3 style='text-align:center; color:gray;'>Machine Learning Project using Streamlit & Scikit-Learn</h3>",
    unsafe_allow_html=True
)

# Load Dataset
data = load_iris()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df['target'] = data.target

# Show Dataset
st.subheader("📊 Dataset")
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
st.sidebar.header("⚙️ Model Parameters")

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

st.subheader("🎯 Accuracy Score")

st.metric(
    label="Model Accuracy",
    value=f"{accuracy:.2f}"
)

# Confusion Matrix
st.subheader("📌 Confusion Matrix")

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
st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
)

st.text(report)

# Decision Tree Visualization
st.subheader("🌳 Decision Tree Visualization")

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
st.subheader("🔍 GridSearchCV")

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

st.write("✅ Best Parameters:")
st.write(grid.best_params_)

st.write("🏆 Best Cross Validation Score:")
st.write(grid.best_score_)