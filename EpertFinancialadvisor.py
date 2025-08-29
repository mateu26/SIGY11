#!/usr/bin/env python3
"""
AI Financial Advisor Web App with AI Learning
---------------------------------------------
Purpose:
  • Online AI expert advisor
  • Users upload bank statements (CSV/PDF)
  • Classifies transactions automatically
  • Learns from data and adapts advice for any income class
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tabula  # For PDF reading

# -------------------------------
# Configuration
# -------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------------------
# Utility Functions
# -------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_bank_statement(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        df = pd.read_csv(file_path, parse_dates=['Date'])
    elif ext == 'pdf':
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        df = pd.concat(tables)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise ValueError("Unsupported file format")
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Category'] = df['Category'].fillna('Other')
    df['Type'] = df['Type'].str.capitalize()
    return df

# -------------------------------
# AI Transaction Classification
# -------------------------------
def classify_transactions(df, model=None, encoder=None):
    if model is None:
        # Create a simple training set from existing data (unsupervised labels could be improved)
        df['Category'] = df['Category'].fillna('Other')
        le = LabelEncoder()
        df['CategoryEncoded'] = le.fit_transform(df['Category'])
        features = pd.get_dummies(df[['Type']])
        model = RandomForestClassifier(n_estimators=100)
        model.fit(features, df['CategoryEncoded'])
    else:
        le = encoder

    features = pd.get_dummies(df[['Type']])
    # Ensure features match trained columns
    for col in model.feature_names_in_:
        if col not in features:
            features[col] = 0
    features = features[model.feature_names_in_]
    predicted = model.predict(features)
    df['CategoryPredicted'] = le.inverse_transform(predicted)
    return df, model, le

# -------------------------------
# Financial Calculations
# -------------------------------
def calculate_cash_flow(df):
    income = df[df['Type'] == 'Income']['Amount'].sum()
    expense = df[df['Type'] == 'Expense']['Amount'].sum()
    net_cash_flow = income - expense
    savings_target = 0.2 * net_cash_flow
    expense_ratios = df.groupby('CategoryPredicted')['Amount'].sum() / net_cash_flow
    return net_cash_flow, savings_target, expense_ratios

def assess_risk(total_debt, total_assets, monthly_expenses, cash):
    rr = total_debt / total_assets if total_assets > 0 else 0
    liquidity_ratio = (cash + 0.0) / monthly_expenses if monthly_expenses > 0 else 0
    credit_score_estimate = 750 - (rr * 100)
    return rr, liquidity_ratio, credit_score_estimate

def recommend_investments(assets, returns, variances):
    expected_returns = np.array(assets) * np.array(returns)
    risk_adjusted = expected_returns / (np.array(variances) + 1e-6)
    allocation = np.exp(risk_adjusted) / np.sum(np.exp(risk_adjusted))
    return allocation

def forecast_balance(current_balance, net_cash_flow, months=6, decay=0.05):
    future_balances = []
    for t in range(1, months+1):
        future_balance = current_balance + net_cash_flow * np.exp(-decay * t)
        future_balances.append(future_balance)
    return future_balances

def generate_advice(df, total_debt, total_assets, monthly_expenses, cash, investment_assets, returns, variances):
    net_cash_flow, savings_target, expense_ratios = calculate_cash_flow(df)
    rr, liquidity_ratio, credit_score_estimate = assess_risk(total_debt, total_assets, monthly_expenses, cash)
    allocation = recommend_investments(investment_assets, returns, variances)
    future_balances = forecast_balance(cash, net_cash_flow)

    advice = {
        'NetCashFlow': net_cash_flow,
        'SavingsTarget': savings_target,
        'ExpenseRatios': expense_ratios.to_dict(),
        'DebtToAssetRatio': rr,
        'LiquidityRatio': liquidity_ratio,
        'CreditScoreEstimate': credit_score_estimate,
        'InvestmentAllocation': dict(zip(investment_assets, allocation)),
        'FutureBalancesNextMonths': future_balances
    }
    return advice

# -------------------------------
# HTML Templates as Strings
# -------------------------------
upload_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload Bank Statement</title>
</head>
<body>
    <h1>AI Financial Advisor</h1>
    <form method="post" enctype="multipart/form-data">
        <p>Upload your bank statement (CSV or PDF):</p>
        <input type="file" name="file">
        <br><br>
        <button type="submit">Upload & Get Advice</button>
    </form>
</body>
</html>
"""

advice_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Financial Advice</title>
</head>
<body>
    <h1>Expert Financial Advice</h1>
    <ul>
        {% for key, value in advice.items() %}
        <li><strong>{{ key }}:</strong> {{ value }}</li>
        {% endfor %}
    </ul>
    <a href="{{ url_for('upload_file') }}">Upload another statement</a>
</body>
</html>
"""

# -------------------------------
# Flask Routes
# -------------------------------
model = None
encoder = None

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            return redirect(url_for('show_advice', filename=filename))
    return render_template_string(upload_html)

@app.route("/advice/<filename>")
def show_advice(filename):
    global model, encoder
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = load_bank_statement(path)

    # AI classification
    df, model, encoder = classify_transactions(df, model, encoder)

    # Default user financial info
    total_debt = float(request.args.get('debt', 5000))
    total_assets = float(request.args.get('assets', 20000))
    monthly_expenses = float(request.args.get('expenses', 1500))
    cash = float(request.args.get('cash', 5000))

    # Example investment options
    investment_assets = ['Stocks', 'Bonds', 'Crypto']
    expected_returns = [0.08, 0.03, 0.12]
    variances = [0.04, 0.01, 0.09]

    advice = generate_advice(df, total_debt, total_assets, monthly_expenses, cash, investment_assets, expected_returns, variances)

    return render_template_string(advice_html, advice=advice)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
