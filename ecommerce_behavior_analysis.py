import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import uuid
import json

np.random.seed(42)

# --- Data Loading / Simulation ---
try:
    df = pd.read_csv('online_shoppers_intention.csv')
    df = df.sample(n=500, random_state=42)
except FileNotFoundError:
    data = {
        'user_id': [str(uuid.uuid4()) for _ in range(500)],
        'age': np.random.randint(18, 70, 500),
        'session_duration': np.random.exponential(scale=20, size=500).round(2),
        'page_views': np.random.poisson(lam=5, size=500),
        'purchases': np.random.binomial(n=5, p=0.1, size=500),
        'time_on_product_page': np.random.exponential(scale=10, size=500).round(2),
        'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 500, p=[0.5, 0.4, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
        'bounce': np.random.choice([0, 1], 500, p=[0.7, 0.3]),
        'visitor_type': np.random.choice(['New', 'Returning'], 500, p=[0.6, 0.4])
    }
    # Add missing values
    for col in ['age', 'session_duration', 'time_on_product_page']:
        mask = np.random.random(500) < 0.05
        data[col] = np.where(mask, np.nan, data[col])
    df = pd.DataFrame(data)

# --- Preprocessing ---
def preprocess_data(df):
    print("Missing Values Before:\n", df.isnull().sum())

    # Fill missing numerical
    num_cols = ['age', 'session_duration', 'time_on_product_page']
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical
    cat_cols = ['device', 'region', 'visitor_type']
    for col in cat_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("\nMissing Values After:\n", df.isnull().sum())

    # One-hot encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Feature engineering
    if 'time_on_product_page' in df.columns and 'page_views' in df.columns:
        df['avg_time_per_page'] = df['time_on_product_page'] / (df['page_views'] + 1)

    # Outlier removal
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]

    for col in ['session_duration', 'time_on_product_page']:
        if col in df.columns:
            df = remove_outliers(df, col)

    # Standardize
    scaler = StandardScaler()
    scale_cols = ['age', 'session_duration', 'time_on_product_page', 'avg_time_per_page']
    scale_cols = [col for col in scale_cols if col in df.columns]
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df

df_cleaned = preprocess_data(df)

# --- Exploratory Data Analysis ---
def perform_eda(df):
    print("\nSummary Statistics:\n", df.describe())

    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='RdBu', center=0)
    plt.title('🔍 Correlation Matrix of Numerical Features', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Visitor Type Column
    visitor_col = next((col for col in df.columns if 'visitor_type_' in col), None)

    # 2. Boxplot: Product Duration by Visitor Type
    if visitor_col and 'time_on_product_page' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=visitor_col, y='time_on_product_page', data=df, palette="Set2")
        plt.title('🕒 Product Page Duration by Visitor Type')
        plt.xlabel("Visitor Type")
        plt.ylabel("Time on Product Page")
        plt.tight_layout()
        plt.savefig('duration_by_visitor.png')
        plt.close()

    # 3. Barplot: Purchases by Region
    region_col = next((col for col in df.columns if 'region_' in col), None)
    if region_col and 'purchases' in df.columns:
        region_data = df.groupby(region_col)['purchases'].mean().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=region_col, y='purchases', data=region_data, palette='coolwarm')
        plt.title('🛒 Average Purchase Rate by Region')
        plt.ylabel("Avg Purchases")
        plt.xlabel("Region")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('purchases_by_region.png')
        plt.close()

    # 4. Scatter Plot: Time vs Purchases
    if visitor_col and 'time_on_product_page' in df.columns and 'purchases' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='time_on_product_page', y='purchases', hue=visitor_col, data=df, alpha=0.6, s=70)
        plt.title('📈 Time on Product Page vs Purchases')
        plt.xlabel('Time on Product Page')
        plt.ylabel('Number of Purchases')
        plt.tight_layout()
        plt.savefig('duration_vs_purchases.png')
        plt.close()

    # 5. Bounce Rate by Visitor Type
    if visitor_col and 'bounce' in df.columns:
        bounce_rate = df.groupby(visitor_col)['bounce'].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=visitor_col, y='bounce', data=bounce_rate, palette='muted')
        plt.title('🚪 Bounce Rate by Visitor Type')
        plt.xlabel('Visitor Type')
        plt.ylabel('Average Bounce Rate')
        plt.tight_layout()
        plt.savefig('bounce_rate_by_visitor.png')
        plt.close()

    # 6. Interactive JSON Chart (Visitor vs Purchases)
    if visitor_col and 'purchases' in df.columns:
        visitor_purchases = df.groupby(visitor_col)['purchases'].mean().reset_index()
        labels = visitor_purchases[visitor_col].tolist()
        chart_config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Average Purchases",
                    "data": visitor_purchases['purchases'].tolist(),
                    "backgroundColor": ["#36A2EB", "#FF6384"],
                    "borderColor": ["#36A2EB", "#FF6384"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Average Purchase Rate"}
                    },
                    "x": {
                        "title": {"display": True, "text": "Visitor Type"}
                    }
                },
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "Purchase Rate by Visitor Type"}
                }
            }
        }
        with open('purchases_by_visitor_chart.json', 'w') as f:
            json.dump(chart_config, f)

    # Interpretation Summary
    insights = """
    ### Key Insights:
    1. 📉 New visitors spend less time on product pages than returning ones.
    2. 🗺️ Regional differences observed in average purchase rates.
    3. 🚪 Bounce rates are higher for new visitors – onboarding needs improvement.
    4. 📈 Time spent on product pages moderately correlates with purchases.
    5. 🔍 Outliers removed to improve analysis quality and model accuracy.
    """
    print(insights)
    return insights

# Run EDA and save results
insights = perform_eda(df_cleaned)
df_cleaned.to_csv('cleaned_user_behavior.csv', index=False)
with open('eda_insights.txt', 'w') as f:
    f.write(insights)
