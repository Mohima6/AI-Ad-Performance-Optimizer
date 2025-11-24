# scripts/simulate_dataset.py

import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
num_ads = 100  # number of ads
audiences = ["Tech Enthusiasts", "Fashion Lovers", "Gamers", "Fitness Fans", "Foodies"]
regions = ["North America", "Europe", "Asia", "South America", "Africa"]
products = ["Smartphone", "Sneakers", "Gaming Laptop", "Protein Powder", "Chocolate Box"]

# Generate synthetic ad data
ads = []
for i in range(num_ads):
    product = random.choice(products)
    audience = random.choice(audiences)
    region = random.choice(regions)
    budget = round(random.uniform(100, 1000), 2)  # budget in $

    # Simulate impressions (more budget → more impressions)
    impressions = int(np.random.normal(loc=budget * 10, scale=50))
    impressions = max(impressions, 50)  # minimum impressions

    # Simulate clicks (CTR depends on audience + randomness)
    ctr = np.random.uniform(0.01, 0.15)  # CTR between 1% and 15%
    clicks = int(impressions * ctr)

    # Simulate conversions (conversion rate depends on clicks)
    conv_rate = np.random.uniform(0.05, 0.25)
    conversions = int(clicks * conv_rate)

    # CTR and ROI
    ctr_calc = clicks / impressions
    roi = round(conversions * 50 / budget, 2)  # assume each conversion = $50

    # Simulate ad text
    ad_text = f"Buy {product} now! Exclusive offer for {audience} in {region}."

    ads.append({
        "Ad ID": f"AD{i + 1:03d}",  # fixed column name
        "Product": product,
        "Audience": audience,
        "Region": region,
        "Budget": budget,            # fixed column name
        "Impressions": impressions,
        "Clicks": clicks,
        "Conversions": conversions,
        "CTR": round(ctr_calc, 3),
        "ROI": roi,
        "Ad Description": ad_text    # fixed column name
    })

# Create DataFrame
df_ads = pd.DataFrame(ads)

# Ensure folder exists
os.makedirs("data/examples", exist_ok=True)

# Save CSV
df_ads.to_csv("data/examples/synthetic_ads_dataset.csv", index=False)
print("Synthetic ad dataset saved to data/examples/synthetic_ads_dataset.csv")
