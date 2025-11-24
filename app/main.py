# app/main.py
import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Try to import the single ML/RL module (ad_optimizer). If it fails, we'll show an error when used.
try:
    from app.models.ML.ad_optimizer import (
        optimize_ads,
        optimize_budget,
        predict_ctr_and_conversions,
        feature_importance,
        generate_insights,
        what_if_simulation
    )
    AD_OPTIMIZER_AVAILABLE = True
except Exception:
    # We'll handle this later without crashing the module import.
    AD_OPTIMIZER_AVAILABLE = False

# -------------------------------
# Utilities: column normalization / validation
# -------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column name variants to the canonical names used throughout the app.
    Canonical names:
      - 'Ad_ID' (internal canonical for ad id)
      - 'Product'
      - 'Audience'
      - 'Region'
      - 'Budget' (float)
      - 'Impressions'
      - 'Clicks'
      - 'Conversions'
      - 'CTR'
      - 'ROI'
      - 'Ad_Description'
    This modifies a copy and returns it.
    """
    df = df.copy()
    col_map = {}

    # map common variants to canonical
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
        else:
            # leave other columns unchanged
            pass

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure canonical Ad_ID exists; if not, create from index
    if "Ad_ID" not in df.columns:
        df.insert(0, "Ad_ID", [f"ROW_{i+1}" for i in range(len(df))])

    # Ensure numeric columns are properly typed (where present)
    for numeric_col in ["Budget", "Impressions", "Clicks", "Conversions", "CTR", "ROI"]:
        if numeric_col in df.columns:
            # coerce to numeric; invalid parse becomes NaN
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    return df

# -------------------------------
# Placeholder / fallback functions (kept so UI works even if ML module missing)
# -------------------------------
def predict_ctr_fallback(df):
    """Return dummy predicted CTR (fallback)."""
    return np.round(np.random.uniform(0.01, 0.15, size=len(df)), 3)

def predict_conversion_fallback(df):
    """Return dummy predicted conversions (fallback)."""
    return np.random.randint(5, 300, size=len(df))

def analyze_text_fallback(df):
    """Return dummy NLP analysis (fallback)."""
    sentiments = ['Positive', 'Neutral', 'Negative']
    return np.random.choice(sentiments, size=len(df))

# -------------------------------
# Scenario Simulation (keeps as-is but normalized)
# -------------------------------
def scenario_simulation(df, budget_increase=0.1):
    """Dummy scenario simulation: increase budget by x%"""
    new_df = df.copy()
    if 'Budget' in df.columns and pd.notnull(new_df['Budget']).any():
        new_df['Optimized_Budget'] = new_df['Budget'] * (1 + budget_increase)
    else:
        new_df['Optimized_Budget'] = np.round(np.random.uniform(100, 200, size=len(df)), 2)
    return new_df

# -------------------------------
# Dashboard (Charts)
# -------------------------------
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
            self.ax.text(0.5, 0.5, "No CTR data to plot", ha='center')
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
            self.ax.text(0.5, 0.5, "No Conversions data to plot", ha='center')
        self.canvas.draw()

    def plot_budget_allocation(self, df):
        self.ax.clear()
        # prefer Optimized_Budget, else Budget; fall back to index if x not available
        if 'Optimized_Budget' in df.columns or 'Budget' in df.columns:
            y_col = 'Optimized_Budget' if 'Optimized_Budget' in df.columns else 'Budget'
            # Determine x column
            if 'Ad_ID' in df.columns:
                x_col = 'Ad_ID'
                plot_df = df.set_index('Ad_ID')
            elif 'Ad ID' in df.columns:
                x_col = 'Ad ID'
                plot_df = df.set_index('Ad ID')
            else:
                # fallback to index
                plot_df = df.copy()
                plot_df.index = [str(i+1) for i in range(len(plot_df))]
            try:
                plot_df[y_col].plot(kind='bar', ax=self.ax)
                self.ax.set_title("Budget Allocation per Ad")
                self.ax.set_ylabel("Budget")
                self.ax.set_xlabel("Ad")
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Unable to plot Budget: {e}", ha='center')
        else:
            self.ax.text(0.5, 0.5, "No Budget data to plot", ha='center')
        self.canvas.draw()

# -------------------------------
# Main Window
# -------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Ad Performance Optimizer")
        self.setGeometry(100, 100, 1200, 800)

        # central layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV Data")
        self.load_btn.clicked.connect(self.load_data)
        self.predict_btn = QPushButton("Predict CTR/Conversion")
        self.predict_btn.clicked.connect(self.run_predictions)
        self.optimize_btn = QPushButton("Optimize Budget (RL)")
        self.optimize_btn.clicked.connect(self.run_optimization)
        self.nlp_btn = QPushButton("Analyze Ad Text")
        self.nlp_btn.clicked.connect(self.run_text_analysis)
        self.simulate_btn = QPushButton("Scenario Simulation")
        self.simulate_btn.clicked.connect(self.run_simulation)
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)

        for b in (self.load_btn, self.predict_btn, self.optimize_btn,
                  self.nlp_btn, self.simulate_btn, self.export_btn):
            btn_layout.addWidget(b)
        self.layout.addLayout(btn_layout)

        # table
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        # dashboard
        self.dashboard = DashboardView()
        self.layout.addWidget(self.dashboard)

        # data storage
        self.ads_df = None

    # -------------------------------
    # Data Handling
    # -------------------------------
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            df = normalize_columns(df)
            self.ads_df = df
            self.display_table()
            QMessageBox.information(self, "Success", f"Data loaded from {file_path}")

            # Automatically run predictions and optimization after upload
            # Use ML module if available, else use fallback
            try:
                self.run_predictions()
                self.run_optimization()
            except Exception as e:
                # don't block load on optimization issues
                QMessageBox.warning(self, "Warning", f"Loaded data but optimization failed: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{str(e)}")

    def display_table(self):
        if self.ads_df is None:
            return
        df = self.ads_df
        # Convert columns to strings for display
        cols = list(df.columns)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            for j, col in enumerate(cols):
                val = row[col]
                # Format floats reasonably
                if isinstance(val, float):
                    display = f"{val:.4f}"
                else:
                    display = str(val)
                self.table.setItem(i, j, QTableWidgetItem(display))
        self.table.resizeColumnsToContents()

    # -------------------------------
    # ML / RL / NLP Integration
    # -------------------------------
    def run_predictions(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return

        try:
            # Use ad_optimizer's prediction if available
            if AD_OPTIMIZER_AVAILABLE:
                # keep canonical columns and call ML prediction
                self.ads_df = predict_ctr_and_conversions(self.ads_df)
            else:
                # fallback: use random dummy predictions
                self.ads_df['Predicted_CTR'] = predict_ctr_fallback(self.ads_df)
                self.ads_df['Predicted_Conversions'] = predict_conversion_fallback(self.ads_df)

            self.display_table()
            self.dashboard.plot_ctr_vs_product(self.ads_df)
            QMessageBox.information(self, "Prediction Done", "CTR and Conversion predictions added.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")

    def run_optimization(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return

        try:
            # Ensure predictions exist
            if 'Predicted_CTR' not in self.ads_df.columns or 'Predicted_Conversions' not in self.ads_df.columns:
                self.run_predictions()

            # Use ad_optimizer's optimize_budget if available
            if AD_OPTIMIZER_AVAILABLE:
                # run full pipeline or only budget allocation depending on your ad_optimizer implementation
                # Here we first ensure ML predictions are present, then call optimize_budget
                # If optimize_ads exists and you want one-call pipeline: use optimize_ads(self.ads_df)
                self.ads_df = optimize_budget(self.ads_df)  # uses df['Predicted_Conversions'] internally
            else:
                # fallback dummy optimization: proportional to predicted conversions
                total_budget = self.ads_df['Budget'].sum() if 'Budget' in self.ads_df.columns else None
                if total_budget is None or np.isnan(total_budget):
                    total_budget = float(len(self.ads_df) * 100.0)
                conv_sum = self.ads_df['Predicted_Conversions'].sum()
                if conv_sum == 0:
                    weights = np.ones(len(self.ads_df)) / len(self.ads_df)
                else:
                    weights = self.ads_df['Predicted_Conversions'] / conv_sum
                self.ads_df['Optimized_Budget'] = (weights * total_budget).round(2)

            # Update UI
            self.display_table()
            self.dashboard.plot_budget_allocation(self.ads_df)
            QMessageBox.information(self, "Optimization Done", "Budget optimization completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"RL Budget Optimization failed:\n{e}")

    def run_text_analysis(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return
        try:
            if AD_OPTIMIZER_AVAILABLE:
                # ad_optimizer doesn't currently implement a text sentiment function; use fallback
                self.ads_df['Ad_Sentiment'] = analyze_text_fallback(self.ads_df)
            else:
                self.ads_df['Ad_Sentiment'] = analyze_text_fallback(self.ads_df)
            self.display_table()
            QMessageBox.information(self, "NLP Analysis Done", "Ad text analysis completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Text analysis failed:\n{e}")

    def run_simulation(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return

        try:
            # Run local scenario simulation then re-run optimization to see new budgets
            self.ads_df = scenario_simulation(self.ads_df, budget_increase=0.1)

            # Optionally re-run budget optimization after simulation
            if AD_OPTIMIZER_AVAILABLE:
                self.ads_df = optimize_budget(self.ads_df)
            else:
                # fallback proportional redistribution (same as in run_optimization)
                total_budget = self.ads_df['Optimized_Budget'].sum() if 'Optimized_Budget' in self.ads_df.columns else \
                               (self.ads_df['Budget'].sum() if 'Budget' in self.ads_df.columns else len(self.ads_df)*100)
                conv_sum = self.ads_df['Predicted_Conversions'].sum() if 'Predicted_Conversions' in self.ads_df.columns else 1
                if conv_sum == 0:
                    weights = np.ones(len(self.ads_df)) / len(self.ads_df)
                else:
                    weights = self.ads_df['Predicted_Conversions'] / conv_sum
                self.ads_df['Optimized_Budget'] = (weights * total_budget).round(2)

            self.display_table()
            self.dashboard.plot_budget_allocation(self.ads_df)
            QMessageBox.information(self, "Simulation Done", "Scenario simulation completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed:\n{e}")

    # -------------------------------
    # Export Report
    # -------------------------------
    def export_report(self):
        if self.ads_df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first!")
            return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "CSV Files (*.csv);;Excel Files (*.xlsx)", options=options
        )
        if not file_path:
            return
        try:
            out_df = self.ads_df.copy()
            # ensure column names are user-friendly on export
            if 'Ad_ID' in out_df.columns and 'Ad ID' not in out_df.columns:
                out_df = out_df.rename(columns={'Ad_ID': 'Ad ID'})
            if file_path.endswith(".csv"):
                out_df.to_csv(file_path, index=False)
            else:
                out_df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Success", f"Report saved at {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report:\n{str(e)}")


# -------------------------------
# Run Application
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

