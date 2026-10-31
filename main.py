import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

print("Loading PhishGuard AI dataset...")

# Download dataset
dataset = fetch_ucirepo(id=967)

# Separate features and target
X = dataset.data.features
y = dataset.data.targets

# Combine into one DataFrame
df = pd.concat([X, y], axis=1)

print("\nDataset successfully loaded!")

# Display dataset information
print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nTarget distribution:")
print(df["label"].value_counts())

# Create plots folder
import os
os.makedirs("plots", exist_ok=True)

# Translate labels for better readability
df["Category"] = df["label"].map({
    0: "Phishing",
    1: "Legitimate"
})

# GRAPH 1: Distribution of URL categories
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Category",
    hue="Category",
    palette=["tomato", "seagreen"],
    legend=False
)

plt.title("Distribution of Phishing and Legitimate URLs")
plt.xlabel("URL Category")
plt.ylabel("Number of URLs")
plt.tight_layout()
plt.savefig("plots/category_distribution.png")
plt.show()


# GRAPH 2: URL length distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="URLLength",
    hue="Category",
    bins=50,
    kde=True,
    element="step"
)

plt.title("URL Length Distribution")
plt.xlabel("URL Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/url_length.png")
plt.show()


# GRAPH 3: HTTPS usage
https_data = df.groupby("Category")["IsHTTPS"].mean().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=https_data,
    x="Category",
    y="IsHTTPS",
    hue="Category",
    palette=["tomato", "seagreen"],
    legend=False
)

plt.title("HTTPS Usage by URL Category")
plt.xlabel("URL Category")
plt.ylabel("Average HTTPS Usage")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/https_usage.png")
plt.show()


print("\nEDA visualizations completed!")
print("Plots saved in the plots folder.")


# ==========================================
# MACHINE LEARNING - RANDOM FOREST
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\nPreparing machine learning model...")

# Select numerical features only
# We exclude the label because that is what we want to predict
X = df.select_dtypes(include=["number"]).drop(
    columns=["label", "URLSimilarityIndex"],
    errors="ignore"
)

# Target variable
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Phishing", "Legitimate"]
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPhishGuard AI model training completed!")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 15 IMPORTANT FEATURES ==========")
print(importance.head(15))

# Visualize feature importance
plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Most Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.show()
