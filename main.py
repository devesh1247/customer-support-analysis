# IMPORT
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATA_PATH    = 'data/customer_support_tickets.csv'
OUTPUT_DIR   = 'outputs'
FIG_DIR      = os.path.join(OUTPUT_DIR, 'figures')
CLEAN_PATH   = os.path.join(OUTPUT_DIR, 'cleaned_tickets.csv')
RANDOM_STATE = 42
TEST_SIZE    = 0.25

os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(FIG_DIR, name + '.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  → saved {path}")

def main():
    # load data
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset:")
    print(" rows:", df.shape[0], "cols:", df.shape[1])
    print(" data types:\n", df.dtypes)
    print(" null counts:\n", df.isnull().sum())
    print(" sample rows:\n", df.head(), "\n")

    # drop unneeded columns
    to_drop = [
        'Ticket ID', 'Customer Name', 'Customer Email',
        'Ticket Subject', 'Ticket Description', 'Resolution'
    ]
    df = df.drop(columns=[c for c in to_drop if c in df])
    print("Dropped text and identifier columns.")

    # parse dates
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
    df['Purchase Year']    = df['Date of Purchase'].dt.year
    df['Purchase Month']   = df['Date of Purchase'].dt.month
    df = df.drop(columns=['Date of Purchase'])

    df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
    df['Time to Resolution']  = pd.to_datetime(df['Time to Resolution'], errors='coerce')

    # compute delays in hours
    df['Response Delay Hrs'] = (
        (df['First Response Time'] - df['First Response Time'].min())
        .dt.total_seconds() / 3600.0
    )
    df['Resolution Hrs'] = (
        (df['Time to Resolution'] - df['First Response Time'])
        .dt.total_seconds() / 3600.0
    )

    df = df.drop(columns=['First Response Time', 'Time to Resolution'])

    # impute missing numeric fields
    for col in ['Customer Age', 'Response Delay Hrs', 'Resolution Hrs']:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"Imputed {col} missing with median {med:.1f}")

    # drop rows missing target
    df = df.dropna(subset=['Customer Satisfaction Rating'])

    # encode categorical fields
    le = LabelEncoder()
    cat_cols = [
        'Customer Gender', 'Product Purchased',
        'Ticket Type', 'Ticket Status',
        'Ticket Priority', 'Ticket Channel'
    ]
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"Encoded {col}")

    # save cleaned data
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_PATH}")

    # exploratory visuals
    sns.set(style="whitegrid")

    plt.figure(figsize=(6,4))
    sns.countplot(x='Customer Satisfaction Rating', data=df)
    plt.title("Satisfaction Rating Counts")
    save_fig('target_distribution')

    plt.figure(figsize=(6,4))
    sns.boxplot(x='Customer Satisfaction Rating', y='Customer Age', data=df)
    plt.title("Customer Age by Satisfaction")
    save_fig('age_vs_satisfaction')

    plt.figure(figsize=(6,4))
    sns.barplot(x='Ticket Priority', y='Resolution Hrs', data=df)
    plt.title("Avg Resolution Time by Priority")
    save_fig('priority_vs_resolution')

    # Top Issues by Product
    # Reload original data to get text columns
    orig_df = pd.read_csv(DATA_PATH)
    plt.figure(figsize=(10, 6))
    top_products = orig_df['Product Purchased'].value_counts().nlargest(5).index
    filtered = orig_df[orig_df['Product Purchased'].isin(top_products)]
    sns.countplot(
        y='Ticket Subject',
        hue='Product Purchased',
        data=filtered,
        order=filtered['Ticket Subject'].value_counts().index[:10]
    )
    plt.title("Top Issues by Product (Top 5 Products)")
    plt.xlabel("Count")
    plt.ylabel("Ticket Subject")
    plt.legend(title="Product", bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig('top_issues_by_product')

    # prepare for modeling
    features = [
        'Customer Age', 'Customer Gender', 'Product Purchased',
        'Ticket Type', 'Ticket Status', 'Ticket Priority',
        'Ticket Channel', 'Purchase Year', 'Purchase Month',
        'Response Delay Hrs', 'Resolution Hrs'
    ]
    target = 'Customer Satisfaction Rating'

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    num_cols = ['Customer Age', 'Response Delay Hrs', 'Resolution Hrs']
    scaler = StandardScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # model training
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    best_model = grid.best_estimator_

    # evaluation
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig('confusion_matrix')

    importances = pd.Series(
        best_model.feature_importances_, index=features
    ).sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    importances.plot(kind='bar')
    plt.title("Feature Importances")
    save_fig('feature_importances')

    print("Pipeline complete. All outputs in", FIG_DIR)

if __name__ == '__main__':
    main()
