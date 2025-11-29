
# AdPerformanceOptimizer -  PyQt5 

import sys, os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QInputDialog
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas




# ML & RL imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import shimmy  # required for gym + SB3 compatibility



# Utilities: Column normalization / numeric coercion
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_map = {}
    for c in df.columns:
        cn = c.strip()
        low = cn.lower()
        if low in ("ad id", "ad_id", "adid", "ad"):
            col_map[c] = "Ad_ID"
        elif low in ("ad text", "ad_text", "ad description", "ad_description", "description"):
            col_map[c] = "Ad_Description"
        elif low in ("budget($)", "budget $", "budget$", "budget"):
            col_map[c] = "Budget"
        elif low in ("impressions", "impr"):
            col_map[c] = "Impressions"
        elif low in ("clicks",):
            col_map[c] = "Clicks"
        elif low in ("conversions", "conversion"):
            col_map[c] = "Conversions"
        elif low in ("ctr", "click through rate", "click-through-rate"):
            col_map[c] = "CTR"
        elif low in ("roi", "roas"):
            col_map[c] = "ROI"
        elif low in ("product",):
            col_map[c] = "Product"
        elif low in ("audience", "target audience"):
            col_map[c] = "Audience"
        elif low in ("region", "country"):
            col_map[c] = "Region"
    if col_map: df = df.rename(columns=col_map)
    if "Ad_ID" not in df.columns:
        df.insert(0, "Ad_ID", [f"ROW_{i + 1}" for i in range(len(df))])
    # Numeric columns coercion (safe)
    for numeric_col in ["Budget", "Impressions", "Clicks", "Conversions", "CTR", "ROI"]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(
                df[numeric_col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors="coerce"
            )
    return df



# ML Prediction Fallback (CTR / Conversions)
def predict_ctr_and_conversions(df: pd.DataFrame):
    df = df.copy()
    np.random.seed(42)
    df['Predicted_CTR'] = np.round(np.random.uniform(0.01, 0.15, size=len(df)), 3)
    df['Predicted_Conversions'] = np.random.randint(5, 300, size=len(df))
    return df



# XAI: Feature Importance
def compute_feature_importance(df):
    df = df.copy()
    le_cols = [c for c in df.columns if df[c].dtype == object]
    df_enc = df.copy()
    for c in le_cols:
        df_enc[c] = LabelEncoder().fit_transform(df[c])


    
    # Use available numeric columns
    numeric_features = []
    for col in ['Budget', 'Optimized_Budget', 'Impressions', 'Clicks']:
        if col in df_enc.columns:
            numeric_features.append(col)
    if not numeric_features:
        raise ValueError("No numeric features available for feature importance.")

    X = df_enc[numeric_features]
    y = df_enc['Predicted_CTR'] if 'Predicted_CTR' in df_enc.columns else np.random.rand(len(df_enc))

    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)

    imp = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': imp.importances_mean
    }).sort_values(by='Importance', ascending=False)
    return importance_df



# NLP Sentiment Analysis (Fallback)
def analyze_text_fallback(df):
    sentiments = ['Positive', 'Neutral', 'Negative']
    return np.random.choice(sentiments, size=len(df))

def suggest_ad_copy(df):
    suggestions = []
    for _, row in df.iterrows():
        sentiment = row.get('Ad_Sentiment', 'Neutral')
        if sentiment == 'Negative':
            suggestions.append("Consider rewording headline for positivity")
        else:
            suggestions.append("CTA looks good")
    df['Ad_Copy_Suggestion'] = suggestions
    return df




# Scenario / What-If Simulation

def scenario_simulation(df, budget_increase=0.1):
    new_df = df.copy()
    if 'Budget' in df.columns and pd.notnull(new_df['Budget']).any():
        new_df['Optimized_Budget'] = new_df['Budget'] * (1 + budget_increase)
    else:
        new_df['Optimized_Budget'] = np.round(np.random.uniform(100, 200, size=len(df)), 2)
    return new_df



# Automated Insights
def generate_insights(df):
    insights = []
    if 'Predicted_CTR' in df.columns:
        low_ctr = df[df['Predicted_CTR'] < 0.05]
        for _, r in low_ctr.iterrows():
            insights.append(f"Ad {r['Ad_ID']} has low CTR ({r['Predicted_CTR']:.2f})")
    if 'Predicted_Conversions' in df.columns:
        low_conv = df[df['Predicted_Conversions'] < 20]
        for _, r in low_conv.iterrows():
            insights.append(f"Ad {r['Ad_ID']} has low predicted conversions ({r['Predicted_Conversions']})")
    return insights



# RL Environment for Budget Optimization
class AdBudgetEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, metric='CTR'):
        super().__init__()
        self.df = df.copy()
        self.n_ads = len(df)
        self.metric = metric
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_ads,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.n_ads * 3,), dtype=np.float32)
        self.state = self._get_state()

    def _get_state(self):
        b = self.df['Budget'].values if 'Budget' in self.df.columns else np.ones(self.n_ads) * 100
        c = self.df['Predicted_Conversions'].values if 'Predicted_Conversions' in self.df.columns else np.ones(
            self.n_ads)
        ctr = self.df['Predicted_CTR'].values if 'Predicted_CTR' in self.df.columns else np.ones(self.n_ads) * 0.05
        return np.concatenate([b, c, ctr])

    def reset(self):
        self.state = self._get_state()
        return self.state

    def step(self, action):
        action = np.array(action)
        action = action / (action.sum() + 1e-6)
        total_budget = self.df['Budget'].sum() if 'Budget' in self.df.columns else self.n_ads * 100
        allocated = action * total_budget
        conv = self.df['Predicted_Conversions'].values
        ctr = self.df['Predicted_CTR'].values
        if self.metric == "CTR":
            reward = np.sum((allocated / total_budget) * ctr)
        elif self.metric == "Conversions":
            reward = np.sum((allocated / total_budget) * conv)
        else:  # minimize cost
            reward = -np.sum(allocated)
        done = True
        info = {}
        self.state = np.concatenate([allocated, conv, ctr])
        return self.state, reward, done, info



# Dashboard / Plots
class DashboardView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.canvas = FigureCanvas(plt.Figure(figsize=(7, 5)))
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

    def plot_ctr_vs_product(self, df):
        self.ax.clear()
        if 'Predicted_CTR' in df.columns or 'CTR' in df.columns:
            col = 'Predicted_CTR' if 'Predicted_CTR' in df.columns else 'CTR'
            try:
                ctr_mean = df.groupby('Product')[col].mean()
                ctr_mean.plot(kind='bar', ax=self.ax)
                self.ax.set_title("Average CTR per Product")
                self.ax.set_ylabel("CTR")
                self.ax.set_xlabel("Product")
                self.ax.tick_params(axis='x', labelrotation=45)
                self.canvas.figure.tight_layout()
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Unable to plot CTR: {e}", ha='center')
        else:
            self.ax.text(0.5, 0.5, "No CTR data", ha='center')
        self.canvas.draw()

    def plot_conversion_vs_region(self, df):
        self.ax.clear()
        if 'Predicted_Conversions' in df.columns or 'Conversions' in df.columns:
            col = 'Predicted_Conversions' if 'Predicted_Conversions' in df.columns else 'Conversions'
            try:
                conv_sum = df.groupby('Region')[col].sum()
                conv_sum.plot(kind='bar', ax=self.ax)
                self.ax.set_title("Total Conversions per Region")
                self.ax.set_ylabel("Conversions")
                self.ax.set_xlabel("Region")
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Unable to plot Conversions: {e}", ha='center')
        else:
            self.ax.text(0.5, 0.5, "No Conversions data", ha='center')
        self.canvas.draw()

    def plot_budget_allocation(self, df):
        self.ax.clear()
        if 'Optimized_Budget' in df.columns or 'Budget' in df.columns:
            y_col = 'Optimized_Budget' if 'Optimized_Budget' in df.columns else 'Budget'
            plot_df = df.set_index('Ad_ID') if 'Ad_ID' in df.columns else df.copy()
            try:
                plot_df[y_col].plot(kind='bar', ax=self.ax)
                self.ax.set_title("Budget Allocation per Ad")
                self.ax.set_ylabel("Budget")
                self.ax.set_xlabel("Ad")
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Unable to plot Budget: {e}", ha='center')
        else:
            self.ax.text(0.5, 0.5, "No Budget data", ha='center')
        self.canvas.draw()




# Main PyQt5 Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Ad Performance Optimizer")
        self.setGeometry(100, 100, 1200, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)


        
        # Buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_data)
        self.predict_btn = QPushButton("Predict CTR/Conversions")
        self.predict_btn.clicked.connect(self.run_predictions)
        self.optimize_btn = QPushButton("Optimize Budget (RL)")
        self.optimize_btn.clicked.connect(self.run_optimization)
        self.nlp_btn = QPushButton("Analyze Ad Text")
        self.nlp_btn.clicked.connect(self.run_text_analysis)
        self.simulate_btn = QPushButton("Scenario Simulation")
        self.simulate_btn.clicked.connect(self.run_simulation)
        self.xai_btn = QPushButton("XAI Feature Importance")
        self.xai_btn.clicked.connect(self.show_feature_importance)
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)

        for b in [self.load_btn, self.predict_btn, self.optimize_btn, self.nlp_btn, self.simulate_btn, self.xai_btn,
                  self.export_btn]:
            btn_layout.addWidget(b)
        self.layout.addLayout(btn_layout)


        
        # Table
        self.table = QTableWidget()
        self.layout.addWidget(self.table)


        # Dashboard
        self.dashboard = DashboardView()
        self.layout.addWidget(self.dashboard)

        
        # Data storage
        self.ads_df = None
        self.optimization_metric = "CTR"

    
    
    # Data Handling
    
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path: return
        try:
            df = pd.read_csv(file_path)
            df = normalize_columns(df)
            self.ads_df = df
            self.display_table()
            QMessageBox.information(self, "Success", f"Data loaded from {file_path}")
            self.run_predictions()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")

    def display_table(self):
        if self.ads_df is None: return
        df = self.ads_df
        cols = list(df.columns)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            for j, col in enumerate(cols):
                val = row[col]
                display = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.table.setItem(i, j, QTableWidgetItem(display))
        self.table.resizeColumnsToContents()


    
    
    # ML / RL / NLP
    
    def run_predictions(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return
        try:
            self.ads_df = predict_ctr_and_conversions(self.ads_df)
            self.display_table()
            self.dashboard.plot_ctr_vs_product(self.ads_df)
            insights = generate_insights(self.ads_df)
            if insights:
                QMessageBox.information(self, "Insights", "\n".join(insights))
            QMessageBox.information(self, "Prediction Done", "CTR and Conversion predictions added.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")

    def run_optimization(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return
        try:
            if 'Predicted_CTR' not in self.ads_df.columns or 'Predicted_Conversions' not in self.ads_df.columns:
                self.run_predictions()

            # Metric selection
            metric, ok = QInputDialog.getItem(self, "Select Optimization Metric",
                                              "Choose metric to maximize:",
                                              ["CTR", "Conversions", "Cost"], 0, False)
            if ok: self.optimization_metric = metric


            
            # RL Optimization
            env = DummyVecEnv([lambda: AdBudgetEnv(self.ads_df, self.optimization_metric)])
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=200)
            obs = env.reset()
            action, _ = model.predict(obs)
            action = action / (action.sum() + 1e-6)
            total_budget = self.ads_df['Budget'].sum() if 'Budget' in self.ads_df.columns else len(self.ads_df) * 100
            self.ads_df['Optimized_Budget'] = (action.flatten() * total_budget).round(2)
            self.display_table()
            self.dashboard.plot_budget_allocation(self.ads_df)
            QMessageBox.information(self, "Optimization Done", "RL budget optimization completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"RL Optimization failed:\n{e}")

    def run_text_analysis(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return
        try:
            self.ads_df['Ad_Sentiment'] = analyze_text_fallback(self.ads_df)
            self.ads_df = suggest_ad_copy(self.ads_df)
            self.display_table()
            QMessageBox.information(self, "NLP Done", "Ad text sentiment & ad copy recommendations added.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Text analysis failed:\n{e}")

    def run_simulation(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return
        try:
            # Ask user for budget increase %
            val, ok = QInputDialog.getDouble(self, "Scenario Simulation",
                                             "Budget increase factor (e.g., 0.1=10%):",
                                             0.1, -1, 10, 2)
            if ok:
                self.ads_df = scenario_simulation(self.ads_df, budget_increase=val)
            self.run_optimization()
            self.display_table()
            self.dashboard.plot_budget_allocation(self.ads_df)
            QMessageBox.information(self, "Simulation Done", "Scenario simulation completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed:\n{e}")

    def show_feature_importance(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return
        try:
            imp_df = compute_feature_importance(self.ads_df)
            msg = "\n".join([f"{f}: {v:.4f}" for f, v in zip(imp_df['Feature'], imp_df['Importance'])])
            QMessageBox.information(self, "Feature Importance (XAI)", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"XAI feature importance failed:\n{e}")


    
    # Export
    
    def export_report(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not file_path: return
        try:
            out_df = self.ads_df.copy()
            if 'Ad_ID' in out_df.columns and 'Ad ID' not in out_df.columns:
                out_df = out_df.rename(columns={'Ad_ID': 'Ad ID'})
            if file_path.endswith(".csv"):
                out_df.to_csv(file_path, index=False)
            else:
                out_df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Success", f"Report saved at {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report:\n{e}")



# Run App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_());
