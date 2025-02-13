## Overview
This project performs an in-depth analysis of synthetic customer data to derive actionable insights. It includes segmenting customers using K-Means clustering and predicting customer engagement via logistic regression. A comprehensive set of visualizations highlights the characteristics of different customer segments and evaluates the predictive model.

## Data Files
- **data/customer_data.csv**: Raw synthetic customer data including CustomerID, Age, Income, Purchase Frequency, and Engagement indicator.
- **results/customer_report.xlsx**: An Excel report summarizing key segmentation metrics and providing the full dataset for further analysis.

## Key Visualizations & Analysis

1. **Customer Segmentation**
   - **Scatter Plot:**  
     - **Description:** Visualizes Age vs. Income with points colored by customer segment.  
     - **Image:**  
       ![Segmentation Scatter Plot](/results/segmentation_scatter.png)
   - **Pairplot:**  
     - **Description:** Explores relationships among Age, Income, and Purchase Frequency across segments.  
     - **Image:**  
       ![Customer Pairplot](/results/customer_pairplot.png)
   - **Boxplots:**  
     - **Description:** Boxplots for Income and Age distributions by segment.  
     - **Images:**  
       ![Income Boxplot](/results/income_boxplot.png)  
       ![Age Boxplot](/results/age_boxplot.png)

2. **Cluster Summary**
   - **Bar Charts:**  
     - **Description:** Summarizes average Age, Income, Purchase Frequency, and Engagement Rate per segment.  
     - **Images:**  
       ![Average Age by Segment](/results/avg_age_by_segment.png)  
       ![Average Income by Segment](/results/avg_income_by_segment.png)  
       ![Average Purchase Frequency by Segment](/results/avg_purchase_frequency_by_segment.png)  
       ![Engagement Rate by Segment](/results/engagement_rate_by_segment.png)

3. **Predictive Modeling of Engagement**
   - **Method:** Logistic regression model predicts customer engagement based on Age, Income, and Purchase Frequency.
   - **Evaluation:** Model accuracy and a detailed classification report are printed during execution.
   - **Visualizations:**  
     - **Confusion Matrix:**  
       ![Confusion Matrix](/results/confusion_matrix.png)  
     - **ROC Curve:**  
       ![ROC Curve](/results/roc_curve.png)

## How to Run

1. **Generate Synthetic Data:**  
   Run the `data_generator.py` script to create `customer_data.csv` in the `data/` folder.
   ```bash
   python data_generator.py