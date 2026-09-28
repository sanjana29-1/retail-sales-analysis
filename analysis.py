"""Retail sales analysis: pandas + SQL (SQLite) + matplotlib.

Run:  python generate_data.py && python analysis.py
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("sales.csv")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2563eb", "#f97316", "#9ca3af"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"])
    # --- Data-quality checks ---
    assert df["order_id"].is_unique, "duplicate order ids"
    assert df.isna().sum().sum() == 0, "missing values found"
    assert (df["revenue"] >= 0).all(), "negative revenue"
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    return df


def sql_queries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Answer business questions with SQL, the way you would against a warehouse."""
    con = sqlite3.connect(":memory:")
    df.to_sql("sales", con, index=False)
    q = {
        "store_summary": """
            SELECT store,
                   COUNT(*)                         AS orders,
                   ROUND(SUM(revenue), 0)           AS revenue,
                   ROUND(AVG(revenue), 2)           AS avg_order_value,
                   ROUND(100.0 * SUM(channel = 'Online') / COUNT(*), 1) AS online_pct
            FROM sales GROUP BY store ORDER BY revenue DESC""",
        "category_summary": """
            SELECT category,
                   ROUND(SUM(revenue), 0)                          AS revenue,
                   ROUND(100.0 * SUM(revenue) / (SELECT SUM(revenue) FROM sales), 1) AS share_pct,
                   ROUND(AVG(discount) * 100, 1)                    AS avg_discount_pct
            FROM sales GROUP BY category ORDER BY revenue DESC""",
        "monthly": """
            SELECT month, ROUND(SUM(revenue), 0) AS revenue, COUNT(*) AS orders
            FROM sales GROUP BY month ORDER BY month""",
        "weekday": """
            SELECT weekday, ROUND(SUM(revenue) / COUNT(DISTINCT date), 0) AS avg_daily_revenue
            FROM sales GROUP BY weekday ORDER BY avg_daily_revenue DESC""",
    }
    return {name: pd.read_sql(sql, con) for name, sql in q.items()}


def charts(r: dict[str, pd.DataFrame]) -> None:
    out = Path(".")  # charts saved next to the README

    m = r["monthly"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(m["month"], m["revenue"] / 1000, marker="o", color=BLUE, linewidth=2)
    peak = m.loc[m["revenue"].idxmax()]
    ax.annotate(f"Peak: {peak['month']}", (peak["month"], peak["revenue"] / 1000),
                textcoords="offset points", xytext=(-120, -4), color=ORANGE)
    ax.set_title("Monthly revenue (AUD, thousands)", loc="left")
    ax.set_ylabel("Revenue ($k)"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out / "chart_monthly_revenue.png"); plt.close(fig)

    c = r["category_summary"].sort_values("revenue")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(c["category"], c["revenue"] / 1000, color=[ORANGE if i == len(c) - 1 else GREY for i in range(len(c))])
    ax.set_title("Revenue by category (AUD, thousands)", loc="left")
    fig.tight_layout(); fig.savefig(out / "chart_category_revenue.png"); plt.close(fig)

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    w = r["weekday"].set_index("weekday").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([d[:3] for d in w["weekday"]], w["avg_daily_revenue"] / 1000,
           color=[ORANGE if d == "Saturday" else BLUE for d in w["weekday"]])
    ax.set_title("Average daily revenue by weekday (AUD, thousands)", loc="left")
    fig.tight_layout(); fig.savefig(out / "chart_weekday_revenue.png"); plt.close(fig)


def insights(df: pd.DataFrame, r: dict[str, pd.DataFrame]) -> list[str]:
    s, c, m, w = r["store_summary"], r["category_summary"], r["monthly"], r["weekday"]
    q4 = m[m["month"] >= "2025-10"]["revenue"].sum() / m["revenue"].sum() * 100
    low_day, high_day = w.iloc[-1], w.iloc[0]
    jun = m[m["month"] == "2025-06"].iloc[0]; may = m[m["month"] == "2025-05"].iloc[0]
    return [
        f"Total revenue: ${df['revenue'].sum():,.0f} across {len(df):,} orders.",
        f"{s.iloc[0]['store']} is the top store (${s.iloc[0]['revenue']:,.0f}); "
        f"average order value ranges ${s['avg_order_value'].min():.2f}-${s['avg_order_value'].max():.2f}.",
        f"{c.iloc[0]['category']} lead revenue at {c.iloc[0]['share_pct']}% share.",
        f"Q4 (Oct-Dec) delivers {q4:.1f}% of annual revenue - plan stock and rosters early.",
        f"{high_day['weekday']} earns {high_day['avg_daily_revenue'] / low_day['avg_daily_revenue']:.1f}x "
        f"the revenue of {low_day['weekday']} - staff accordingly.",
        f"June (EOFY sale) had {jun['orders'] / may['orders'] - 1:+.0%} more orders than May but "
        f"{jun['revenue'] / may['revenue'] - 1:+.0%} revenue - deep discounts drove volume, not value.",
    ]


if __name__ == "__main__":
    df = load()
    results = sql_queries(df)
    for name, table in results.items():
        print(f"\n=== {name} ===\n{table.to_string(index=False)}")
    charts(results)
    print("\n=== Key insights ===")
    for line in insights(df, results):
        print("-", line)
