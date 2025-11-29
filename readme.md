
# AI Ad Performance Optimizer

**PyQt5: for Ad Analytics, Optimization, and Insights**

---

## **Overview**

The **AI Ad Performance Optimizer** is a comprehensive desktop application built with **Python, PyQt5, scikit-learn, Stable-Baselines3 (RL), and Matplotlib**. It helps digital marketers and advertisers analyze ad campaigns, predict CTR & conversions, optimize ad budgets, and generate actionable insights—all from a single CSV dataset.

This tool integrates **ML prediction**, **RL-based budget optimization**, **NLP sentiment analysis**, and **interactive visualizations** to make ad performance management smarter and faster.

---

## **Key Features**

### **Data Handling**

* Load ad campaign data from CSV files.
* Automatically normalize and clean columns (Ad_ID, Budget, CTR, Conversions, etc.).
* Handle numeric coercion for financial and performance metrics.

### **Machine Learning**

* Predict **CTR** (Click Through Rate) and **Conversions** for ads using ML fallback.
* Generate **automated insights** highlighting low-performing ads.

### **Explainable AI (XAI)**

* Compute **feature importance** using Random Forest models.
* Identify which ad metrics (e.g., Budget, Impressions, Clicks) influence performance most.

### **Reinforcement Learning (RL)**

* Optimize **ad budgets** across campaigns using **PPO (Proximal Policy Optimization)**.
* Users can select the metric to maximize: CTR, Conversions, or minimize Cost.
* View optimized budget allocation in interactive plots.

### **NLP Ad Analysis**

* Analyze ad text sentiment (Positive, Neutral, Negative).
* Provide **ad copy suggestions** to improve engagement.

### **Scenario Simulation**

* Simulate "what-if" scenarios like **budget increases**.
* Evaluate impact on predicted CTR and conversions.

### **Interactive Dashboard**

* Visualize metrics like CTR per Product, Conversions per Region, Budget allocation.
* Includes **zoom/pan toolbar** and **save plot** functionality.

### **Export Reports**

* Export processed data and insights to **CSV or Excel**.
* Includes predicted metrics, optimized budget, sentiment analysis, and suggestions.

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/Mohima6/AI-Ad-Performance-Optimizer.git
cd AI-Ad-Performance-Optimizer
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

**Requirements**: Python ≥3.10, PyQt5, pandas, numpy, matplotlib, scikit-learn, stable-baselines3, gym, shimmy

3. Run the application:

```bash
python app/main.py
```

---

## **Usage**

1. **Load CSV**: Click "Load CSV" and select your ad campaign dataset.
2. **Predict Performance**: Click "Predict CTR/Conversions" to generate predictions.
3. **Optimize Budget**: Click "Optimize Budget (RL)" and choose a metric to maximize.
4. **Text Analysis**: Click "Analyze Ad Text" for sentiment and copy suggestions.
5. **Scenario Simulation**: Click "Scenario Simulation" to test budget changes.
6. **XAI Feature Importance**: Click "XAI Feature Importance" to see which features matter most.
7. **Export Report**: Click "Export Report" to save your processed dataset.


## **Credits**

* Python, PyQt5, pandas, numpy, matplotlib, scikit-learn, stable-baselines3, gym
* Inspired by the need for smarter **ad budget optimization and campaign insights**

---

## **Problem Statement & Motivation**

* Running effective digital advertising campaigns is challenging because:

* Marketers often have hundreds of ads across multiple platforms, making manual analysis time-consuming.

* Budgets are limited, and inefficient allocation can drastically reduce ROI.

* Identifying high-performing ads and understanding what influences performance is difficult without proper analytics.

* Ad copy and messaging often lack data-driven improvements, which can hurt engagement.

## **This project solves these problems by-**

* Automatically predicting CTR and conversions for all ads in a dataset.

* Providing RL-based budget optimization to maximize returns from limited resources.

* Offering explainable insights (XAI) on what metrics drive ad performance.

* Suggesting improvements for ad copy based on sentiment analysis.

* Allowing interactive visualization and scenario simulations for better decision-making.

In short: It helps marketers save time, improve ROI, and make data-driven decisions with ease.

---

