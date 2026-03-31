import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Set page config
st.set_page_config(page_title="Airbnb Paris Dashboard", page_icon="🏠", layout="wide")

# Title
st.title("🏠 Airbnb Paris Dashboard")
st.markdown("Explore Airbnb listings data for Paris")

# Load data
@st.cache_data
def load_data():
    # Use the new parquet file
    df = pd.read_parquet('../data/cleaned/paris_airbnb_cleaned.parquet')
    return df

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
    st.subheader("📊 Dataset Overview")
    st.write(f"Total listings: {len(filtered_df):,}")
    st.write(f"Average price: €{filtered_df['price'].mean():.2f}")
    st.write(f"Median price: €{filtered_df['price'].median():.2f}")

with col2:
    st.subheader("📈 Price Statistics")
    st.metric("Min Price", f"€{filtered_df['price'].min()}")
    st.metric("Max Price", f"€{filtered_df['price'].max()}")

# Charts
st.header("📊 Visualizations")

# Price Distribution
st.subheader("Price Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['price'], bins=30, kde=True, color='#2ecc71', ax=ax)
ax.set_title('Distribution of Airbnb Prices in Paris', fontsize=15)
ax.set_xlabel('Price per Night (€)')
ax.set_ylabel('Number of Listings')
st.pyplot(fig)

# Neighbourhood Distribution
st.subheader("Listings by Neighbourhood")
fig, ax = plt.subplots(figsize=(12, 8))
neighbourhood_counts = filtered_df['neighbourhood'].value_counts().head(20)
sns.barplot(x=neighbourhood_counts.values, y=neighbourhood_counts.index, palette='magma', ax=ax)
ax.set_title('Top 20 Neighbourhoods by Listing Count', fontsize=15)
ax.set_xlabel('Number of Listings')
st.pyplot(fig)

# Bedrooms vs Price
st.subheader("Bedrooms vs Price by Room Type")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='bedrooms', y='price', hue='room_type', alpha=0.6, ax=ax)
ax.set_title('How Bedrooms and Room Type Impact Price', fontsize=15)
ax.set_xlabel('Number of Bedrooms')
ax.set_ylabel('Price per Night (€)')
st.pyplot(fig)

# Data table
st.header("📋 Sample Data")
st.dataframe(filtered_df.head(100))

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit | Data: Airbnb Paris Listings")