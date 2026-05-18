# Bank Customer Churn Analysis & Prediction Dashboard

An interactive, end-to-end data analysis and machine learning dashboard built with **Streamlit** and **Plotly**. This project explores a bank customer dataset to understand why customers are churning (leaving the bank), profiles highly engaged customers, and uses a Logistic Regression model to predict the churn probability of new or existing customers in real time.

## 🌟 Features

- **Dynamic Theming**: Custom-built Light and Dark modes seamlessly applied across all Streamlit components and Plotly charts.
- **Customer Overview**: High-level KPIs, demographic breakdowns (Age, Geography, Gender), and financial distribution analysis.
- **Churn Analysis**: Deep-dive into factors driving customer attrition using statistical insights.
- **Real-Time Churn Predictor**: An interactive simulator that takes in customer demographics and financial indicators to predict their exact likelihood of churning, complete with a key-driver analysis (waterfall chart) explaining *why* the model made that decision.
- **Engagement Profiling**: Identifies "Strong Engagement" and "Potential Active" customers for targeted retention and cross-selling campaigns.

## 🛠️ Technology Stack

- **Python 3.9+**
- **Streamlit**: Web application framework
- **Plotly Express / Graph Objects**: Interactive data visualizations
- **Scikit-Learn**: Machine learning (Logistic Regression, MinMaxScaler)
- **Pandas & NumPy**: Data processing and feature engineering
- **SciPy**: Statistical hypothesis testing

## 📊 The Data Science Process

The complete data pipeline is documented in `bank_churn_analysis.ipynb`:
1. **Data Cleaning**: Handled missing values, standardized column names, and cleaned financial strings.
2. **EDA & Hypothesis Testing**: Conducted Chi-Square tests for categorical variables and Mann-Whitney U / T-tests for continuous variables to validate churn drivers.
3. **Feature Engineering**: Applied one-hot encoding, binning (e.g., Age Groups, Salary Groups), and MinMax scaling.
4. **Machine Learning**: Trained a `LogisticRegression` model evaluated using Stratified K-Fold cross-validation. Optimized the decision boundary using a Precision-Recall curve to find the threshold (59%) that maximizes the F1-Score.
