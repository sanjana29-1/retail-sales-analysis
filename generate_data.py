"""Generate a synthetic fashion-retail transactions dataset.

The data is SYNTHETIC (randomly generated with realistic patterns) so the
project can be shared publicly without exposing any real business data.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

STORES = {"Parramatta": 1.25, "Sydney CBD": 1.40, "Chatswood": 1.00}
CATEGORIES = {  # category: (avg price AUD, relative popularity)
    "Dresses": (89, 0.18), "Tops": (45, 0.26), "Jeans": (79, 0.14),
    "Outerwear": (149, 0.08), "Accessories": (29, 0.20), "Shoes": (119, 0.14),
}
CHANNELS = ["In-store", "Online"]
MONTH_SEASONALITY = {1: 0.95, 2: 0.80, 3: 0.85, 4: 0.90, 5: 1.00, 6: 1.10,
                     7: 1.05, 8: 0.90, 9: 0.92, 10: 1.00, 11: 1.30, 12: 1.55}
WEEKDAY_FACTOR = {0: 0.75, 1: 0.70, 2: 0.80, 3: 1.05, 4: 1.10, 5: 1.45, 6: 1.20}


def generate(start="2025-01-01", end="2025-12-31", base_orders=55) -> pd.DataFrame:
    rows, order_id = [], 100000
    cats = list(CATEGORIES)
    probs = np.array([CATEGORIES[c][1] for c in cats]); probs /= probs.sum()
    for day in pd.date_range(start, end, freq="D"):
        for store, store_factor in STORES.items():
            lam = base_orders * store_factor * MONTH_SEASONALITY[day.month] * WEEKDAY_FACTOR[day.dayofweek]
            for _ in range(RNG.poisson(lam)):
                order_id += 1
                cat = RNG.choice(cats, p=probs)
                price = max(5.0, RNG.normal(CATEGORIES[cat][0], CATEGORIES[cat][0] * 0.15))
                qty = int(RNG.choice([1, 1, 1, 2, 2, 3]))
                # Bigger discounts during the end-of-financial-year (June) and Boxing Day sales
                disc_mean = 0.25 if (day.month == 6 or (day.month == 12 and day.day >= 26)) else 0.05
                discount = float(np.clip(RNG.normal(disc_mean, 0.05), 0, 0.5))
                channel = RNG.choice(CHANNELS, p=[0.7, 0.3])
                rows.append({
                    "order_id": order_id, "date": day.date(), "store": store,
                    "channel": channel, "category": cat, "quantity": qty,
                    "unit_price": round(price, 2), "discount": round(discount, 2),
                    "revenue": round(qty * price * (1 - discount), 2),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("sales.csv", index=False)
    print(f"Wrote sales.csv with {len(df):,} rows")
