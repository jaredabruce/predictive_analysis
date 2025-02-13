import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def main():
    # Ensure required directories exist
    os.makedirs('results', exist_ok=True)
    
    # -----------------------------------
    # 1. Load and Preprocess Data
    # -----------------------------------
    data = pd.read_csv('data/customer_data.csv')
    # Use forward fill for missing values
    data.ffill(inplace=True)
    
    print("Sample of Customer Data:")
    print(data.head())
    
    # -----------------------------------
    # 2. Customer Segmentation with K-Means Clustering
    # -----------------------------------
    clustering_features = data[['Age', 'Income', 'Purchase_Frequency']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Segment'] = kmeans.fit_predict(clustering_features)
    
    print("Customer Data with Segments:")
    print(data.head())
    
    # Visualization A: Scatter Plot of Age vs. Income by Segment
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Income', y='Age', hue='Segment', data=data, palette='viridis', s=100, alpha=0.7)
    plt.title('Customer Segmentation: Age vs Income')
    plt.xlabel('Income')
    plt.ylabel('Age')
    plt.savefig('results/segmentation_scatter.png')
    plt.close()
    
    # Visualization B: Pairplot of Customer Features by Segment
    pairplot = sns.pairplot(data, vars=['Age', 'Income', 'Purchase_Frequency'], hue='Segment', palette='viridis')
    pairplot.fig.suptitle("Pairplot of Customer Features by Segment", y=1.02)
    pairplot.savefig('results/customer_pairplot.png')
    plt.close()
    
    # Visualization C: Boxplots for Income and Age by Segment
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Segment', y='Income', data=data, palette='viridis')
    plt.title('Income Distribution by Customer Segment')
    plt.savefig('results/income_boxplot.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Segment', y='Age', data=data, palette='viridis')
    plt.title('Age Distribution by Customer Segment')
    plt.savefig('results/age_boxplot.png')
    plt.close()
    
    # Visualization D: Cluster Summary Bar Charts
    cluster_summary = data.groupby('Segment').agg({
        'Age': 'mean',
        'Income': 'mean',
        'Purchase_Frequency': 'mean',
        'Engaged': 'mean'
    }).reset_index()
    cluster_summary.rename(columns={'Engaged': 'Engagement_Rate'}, inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Segment', y='Age', data=cluster_summary, palette='viridis')
    plt.title('Average Age by Customer Segment')
    plt.savefig('results/avg_age_by_segment.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Segment', y='Income', data=cluster_summary, palette='viridis')
    plt.title('Average Income by Customer Segment')
    plt.savefig('results/avg_income_by_segment.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Segment', y='Purchase_Frequency', data=cluster_summary, palette='viridis')
    plt.title('Average Purchase Frequency by Customer Segment')
    plt.savefig('results/avg_purchase_frequency_by_segment.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Segment', y='Engagement_Rate', data=cluster_summary, palette='viridis')
    plt.title('Engagement Rate by Customer Segment')
    plt.ylim(0, 1)
    plt.savefig('results/engagement_rate_by_segment.png')
    plt.close()
    
    # -----------------------------------
    # 3. Predictive Modeling: Logistic Regression for Engagement
    # -----------------------------------
    features = data[['Age', 'Income', 'Purchase_Frequency']]
    target = data['Engaged']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression Accuracy: {accuracy:.2f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualization E: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Engaged', 'Engaged'], yticklabels=['Not Engaged', 'Engaged'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Visualization F: ROC Curve for Logistic Regression
    from sklearn.metrics import roc_curve, auc
    y_proba = log_reg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    plt.close()
    
    # -----------------------------------
    # 4. Automated Excel Reporting
    # -----------------------------------
    with pd.ExcelWriter('results/customer_report.xlsx') as writer:
        cluster_summary.to_excel(writer, sheet_name='Segment Summary', index=False)
        data.to_excel(writer, sheet_name='Full Data', index=False)
    
    print("Excel report saved to results/customer_report.xlsx")

if __name__ == '__main__':
    main()
