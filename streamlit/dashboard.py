import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Set page config
st.set_page_config(page_title="Airbnb Paris Dashboard", page_icon="🏠", layout="wide")

# Title
st.title("🏠 Airbnb Paris Dashboard")
st.markdown("Explore Airbnb listings data for Paris")

# Load data
@st.cache_data
def load_data():
    current_dir = Path(__file__).parent
    
    file_path = (current_dir / "paris_airbnb_cleaned.parquet").resolve()
    
    return pd.read_parquet(file_path)

viz_df = load_data()

# Sidebar
st.sidebar.header("Filters")

# Price range filter
price_range = st.sidebar.slider(
    "Price Range (€)",
    min_value=int(viz_df['price'].min()),
    max_value=int(viz_df['price'].max()),
    value=(int(viz_df['price'].min()), int(viz_df['price'].max()))
)

# Room type filter
room_types = viz_df['room_type'].unique()
selected_room_types = st.sidebar.multiselect(
    "Room Types",
    options=room_types,
    default=room_types
)

# Neighbourhood filter
neighbourhoods = sorted(viz_df['neighbourhood'].unique())
selected_neighbourhoods = st.sidebar.multiselect(
    "Neighbourhoods",
    options=neighbourhoods,
    default=neighbourhoods[:5] if len(neighbourhoods) > 5 else neighbourhoods
)

# Filter data
filtered_df = viz_df[
    (viz_df['price'] >= price_range[0]) &
    (viz_df['price'] <= price_range[1]) &
    (viz_df['room_type'].isin(selected_room_types)) &
    (viz_df['neighbourhood'].isin(selected_neighbourhoods))
]

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" Dataset Overview")
    st.write(f"Total listings: {len(filtered_df):,}")
    st.write(f"Average price: €{filtered_df['price'].mean():.2f}")
    st.write(f"Median price: €{filtered_df['price'].median():.2f}")

with col2:
    st.subheader(" Price Statistics")
    st.metric("Min Price", f"€{filtered_df['price'].min()}")
    st.metric("Max Price", f"€{filtered_df['price'].max()}")

# Charts
st.header(" Visualizations")

# Price distribution tarple (linear + log)
col_chart_price_a, col_chart_price_b = st.columns(2)
with col_chart_price_a:
    st.subheader("Price Distribution (linear)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(filtered_df['price'], bins=60, color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.axvline(filtered_df['price'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean €{filtered_df['price'].mean():.0f}")
    ax.axvline(filtered_df['price'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"Median €{filtered_df['price'].median():.0f}")
    ax.set_title('Price Distribution (filtered data)', fontsize=13)
    ax.set_xlabel('Price (€/night)')
    ax.set_ylabel('Number of listings')
    ax.legend()
    st.pyplot(fig)

with col_chart_price_b:
    st.subheader("Price Distribution (log scale)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log1p(filtered_df['price']), bins=60, color='#55A868', edgecolor='white', linewidth=0.5)
    ax.set_title('Price Distribution — log scale', fontsize=13)
    ax.set_xlabel('log(Price + 1)')
    ax.set_ylabel('Number of listings')
    st.pyplot(fig)

# Neighbourhood and room insights
st.subheader("Top 20 Neighbourhoods by Listing Count")
neighbourhood_counts = filtered_df['neighbourhood'].value_counts().head(20)
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(neighbourhood_counts.index, neighbourhood_counts.values, color='#4C72B0', edgecolor='white', linewidth=0.5)
for bar in bars:
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2, f'{int(bar.get_width())}', va='center', fontsize=9)
ax.set_title('Top 20 Neighbourhoods by Listing Count', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of listings')
ax.set_ylabel('Neighbourhood')
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig)

st.subheader("Room Type Distribution")
room_counts = filtered_df['room_type'].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2'][:len(room_counts)]
ax.pie(room_counts, labels=room_counts.index, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops=dict(width=0.4), textprops={'fontsize': 11})
ax.set_title('Room Type Distribution (filtered data)', fontsize=13, fontweight='bold', pad=20)
st.pyplot(fig)

# Neighborhood price variance
st.subheader('Price by Neighbourhood — Top 15 by Median Price')
top15 = (filtered_df.groupby('neighbourhood')['price'].median().sort_values(ascending=False).head(15).index)
plot_data = filtered_df[filtered_df['neighbourhood'].isin(top15)]
fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(data=plot_data, x='neighbourhood', y='price', order=top15, palette='Blues_r', width=0.6, flierprops=dict(marker='o', markersize=2, alpha=0.3), ax=ax)
ax.set_title('Price distribution by neighbourhood — top 15 (by median)', fontsize=13, fontweight='bold')
ax.set_xlabel('Neighbourhood')
ax.set_ylabel('Price (€/night)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Amenities vs price
st.subheader('Amenities Count vs Price (with trend line)')
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(filtered_df['amenities_count'], filtered_df['price'], alpha=0.15, s=10, c=filtered_df['bedrooms'], cmap='viridis')
if filtered_df['amenities_count'].notna().sum() > 2:
    z = np.polyfit(filtered_df['amenities_count'].dropna(), filtered_df.loc[filtered_df['amenities_count'].notna(), 'price'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(filtered_df['amenities_count'].min(), filtered_df['amenities_count'].max(), 200)
    ax.plot(x_line, p(x_line), color='red', linewidth=2, label='Trend line')
    ax.legend()
plt.colorbar(scatter, ax=ax, label='Bedrooms')
ax.set_title('Amenities count vs price (filtered data)', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of amenities')
ax.set_ylabel('Price (€/night)')
plt.tight_layout()
st.pyplot(fig)

# Correlation matrix
st.subheader('Correlation matrix — numeric variables')
numeric_vars = ['price', 'bedrooms', 'amenities_count', 'review_scores_rating', 'accommodates', 'minimum_nights']
if set(numeric_vars).issubset(filtered_df.columns):
    corr = filtered_df[numeric_vars].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation matrix — numeric variables', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Strongest correlations with price (filtered):', corr['price'].drop('price').sort_values(ascending=False).round(3))
else:
    st.warning('Some numeric variables for correlation are unavailable in the data.')

# Data table
st.header("Sample Data")
st.dataframe(filtered_df.head(100))

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit | Data: Airbnb Paris Listings")