import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy import stats as scipy_stats
# Set page config
st.set_page_config(page_title="Airbnb Paris Dashboard", page_icon="🏠", layout="wide")

# Title
st.title("🏠 Airbnb Paris Dashboard")
st.markdown("Explore Airbnb listings data for Paris markit place ")

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


import streamlit as st

# Title for the section
st.subheader(" Paris Airbnb Market Overview")

# 1. Create a row of 5 columns for your 5 KPIs
col1,  col6 ,col7, col2, col3, col4, col5 , = st.columns(7)

# 2. Assign each KPI to a column using st.metric
with col1:
    st.metric(label="Total Listings", value=f"{len(viz_df):,}")

with col2:
    st.metric(label="Average Price", value=f"€{viz_df['price'].mean():.2f}")

with col3:
    st.metric(label="Median Price", value=f"€{viz_df['price'].median():.0f}")
with col6:
    st.metric(label="Lowest Price", value=f"€{viz_df['price'].min()}")
with col7:
    st.metric(label="Highest Price", value=f"€{viz_df['price'].max()}")
with col4:
    st.metric(label="Neighbourhoods", value=viz_df['neighbourhood'].nunique())

with col5:
    st.metric(label="Avg Amenities", value=f"{viz_df['amenities_count'].mean():.1f}")

# Add a divider to separate the KPIs from your charts
st.divider()

# Charts

# --- 1. Header & Global Style ---
st.title(" Data Exploration & Visual Insights")
st.markdown("---")

# --- 2. Price Distribution Section (Univariate) ---
st.header("1. Price Analysis")
col_a, col_b = st.columns([2, 1]) # Column 'a' is twice as wide

with col_a:
    # Linear and Log Price Distribution
    tabs = st.tabs(["Linear Scale", "Log Scale"])
    
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['price'], bins=60, color='#4C72B0', kde=True, ax=ax)
        ax.axvline(filtered_df['price'].mean(), color='red', linestyle='--', label=f"Mean: €{filtered_df['price'].mean():.0f}")
        ax.axvline(filtered_df['price'].median(), color='orange', linestyle='--', label=f"Median: €{filtered_df['price'].median():.0f}")
        ax.set_title('Price Distribution (Filtered Data)')
        ax.legend()
        st.pyplot(fig)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(np.log1p(filtered_df['price']), bins=60, color='#55A868', kde=True, ax=ax)
        ax.set_title('Price Distribution (Log Scale)')
        st.pyplot(fig)

with col_b:
    st.subheader("Price Insights")
    # Dynamic Calculation
    skewness = filtered_df['price'].skew()
    st.info(f"""
    **Key Observations:**
    * **Median Price:** €{filtered_df['price'].median():.0f} per night.
    * **Skewness:** {skewness:.2f}. The data is right-skewed, meaning a few high-priced luxury listings pull the mean up.
    * **Log Scale:** Transforming the price helps normalize the distribution for better statistical modeling.
    """)

st.markdown("---")

# --- 3. Neighborhood & Room Types (Categorical) ---
st.header("2. Location Analysis")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Top 20 Neighbourhoods by Listing Volume")
    n_counts = filtered_df['neighbourhood'].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=n_counts.values, y=n_counts.index, palette='viridis', ax=ax)
    ax.set_title('Top Listings by Volume (Filtered Data)')
    # Adding data labels for better readability
    for i, v in enumerate(n_counts.values):
        ax.text(v + 3, i, str(int(v)), va='center', fontsize=9)
    st.pyplot(fig)

with col_b:
    st.subheader("Neighbourhood Supply Insights")
    # Dynamic Calculations
    top1_name = n_counts.index[0]
    top1_count = n_counts.iloc[0]
    top5_share = (filtered_df['neighbourhood'].value_counts().head(5).sum() / len(filtered_df)) * 100
    
    st.info(f"""
    **--- Dashboard Insight ---**
    Airbnb supply in Paris is highly concentrated. The top 5 neighbourhoods 
    account for **{top5_share:.1f}%** of all listings in your current selection. 
    
    **{top1_name}** dominates the market with **{top1_count}** listings, reflecting 
    significant central demand and market saturation in this arrondissement.
    """)

st.markdown("---")


st.header("3. Property Type Analysis")
col_a, col_b = st.columns([2, 1])

with col_a:
    # 1. Prepare Data
    room_counts = filtered_df['room_type'].value_counts()
    room_pct = (room_counts / len(filtered_df)) * 100
    
    # 2. Square, compact figure size
    fig, ax = plt.subplots(figsize=(4, 4)) 
    
    # 3. Clean labels for the plot (Mapping to shorter names)
    label_map = {
        'Entire place': 'Entire Place',
        'Private room': 'Private',
        'Hotel room': 'Hotel',
        'Shared room': 'Shared'
    }
    plot_labels = [label_map.get(x, x) for x in room_counts.index]
    
    # 4. Donut Chart with improved spacing
    wedges, texts, autotexts = ax.pie(
        room_counts, 
        labels=plot_labels, 
        autopct='%1.0f%%', 
        startangle=220, 
        pctdistance=0.75,    # Numbers sit inside the ring
        labeldistance=1.15,  # Labels sit clearly outside the ring
        colors=sns.color_palette('pastel'),
        wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 2}
    )
    
    # Format text for clarity
    plt.setp(autotexts, size=9, weight="bold")
    plt.setp(texts, size=8)
    fig.tight_layout()
    st.pyplot(fig)

with col_b:
    # --- Dynamic Calculations ---
    top_room = room_counts.index[0]
    room_price = filtered_df.groupby('room_type')['price'].median()
    
    # Handle naming variation ('Entire place' vs others)
    entire_median = room_price.get('Entire place', 0)
    private_median = room_price.get('Private room', 0)
    
    # Calculate Premium
    premium_val = 0
    if private_median > 0:
        premium_val = ((entire_median / private_median) - 1) * 100

    st.caption(" **Occupancy Insights**")
    st.info(f"""
    **--- Dashboard Insight ---**
    **{top_room}** dominates the market at **{room_pct.iloc[0]:.0f}%** of listings.
    
    **Price Premium:** Entire homes (**€{entire_median:.0f}**) command a **{premium_val:.0f}%** premium over private rooms (**€{private_median:.0f}**). 
    
    The low share of shared rooms confirms a strong traveler preference for privacy in Paris.
    """)

st.markdown("---")

# --- 4. Price Variance & Features (Bi/Multivariate) ---
st.header("3. Factors Driving Price")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Price Variance by Top 15 Neighbourhoods")
    
    # Identify Top 15 by Median
    top15 = filtered_df.groupby('neighbourhood')['price'].median().sort_values(ascending=False).head(15).index
    plot_data = filtered_df[filtered_df['neighbourhood'].isin(top15)]
    
    # Slightly smaller height to keep the "compact" feel
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Horizontal boxplot for better readability of names
    sns.boxplot(
        data=plot_data, 
        y='neighbourhood', 
        x='price', 
        order=top15, 
        palette='Blues_r', 
        ax=ax,
        fliersize=2,      # Smaller outlier dots
        linewidth=1       # Thinner lines for a cleaner look
    )
    
    ax.set_title('Price Distribution (Top 15 by Median)', fontsize=10)
    ax.set_xlabel('Price (€/night)', fontsize=9)
    ax.set_ylabel('', fontsize=9)
    ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    # --- Dynamic Calculations ---
    n_stats = (filtered_df.groupby('neighbourhood')['price']
                 .agg(['median', 'std'])
                 .sort_values('median', ascending=False))
    
    most_expensive = n_stats.index[0]
    least_expensive = n_stats.index[-1]
    price_gap = n_stats.loc[most_expensive, 'median'] - n_stats.loc[least_expensive, 'median']
    highest_variance = n_stats['std'].idxmax()

    st.caption("🗺️ **Geographic Price Gap**")
    st.info(f"""
    **--- Dashboard Insight ---**
    There is a **€{price_gap:.0f}/night** gap between **{most_expensive}** and **{least_expensive}**.
    
    **Volatility:** **{highest_variance}** shows the highest price variance (std = €{n_stats.loc[highest_variance, 'std']:.0f}), suggesting it’s a diverse hub where budget and luxury coexist. 
    
    Location remains the primary anchor for Paris pricing.
    """)

st.markdown("---")
# Amenities vs Price
st.header("5. Impact of Amenities on Pricing")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Amenities Count vs. Price")
    
    # Clean data for the trend line
    clean_scatter = filtered_df[['amenities_count', 'price']].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Scatter plot with low alpha to handle overplotting
    sns.regplot(
        data=clean_scatter, 
        x='amenities_count', 
        y='price', 
        scatter_kws={'alpha':0.2, 's':10, 'color':'#4C72B0'}, 
        line_kws={'color':'red', 'lw':2},
        ax=ax
    )
    
    ax.set_title('Correlation: Amenities Volume vs. Nightly Rate', fontsize=10)
    ax.set_xlabel('Number of Amenities', fontsize=9)
    ax.set_ylabel('Price (€/night)', fontsize=9)
    ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    # --- Dynamic Statistical Calculations ---
    r, p_val = scipy_stats.pearsonr(clean_scatter['amenities_count'], clean_scatter['price'])
    
    # Quantile Calculations
    low_q = filtered_df['amenities_count'].quantile(0.25)
    high_q = filtered_df['amenities_count'].quantile(0.75)
    
    low_med = filtered_df[filtered_df['amenities_count'] <= low_q]['price'].median()
    high_med = filtered_df[filtered_df['amenities_count'] >= high_q]['price'].median()
    
    # Avoid division by zero
    price_lift = ((high_med / low_med) - 1) * 100 if low_med > 0 else 0

    st.caption("🧪 **Statistical Inference**")
    st.info(f"""
    **--- Dashboard Insight ---**
    There is a **{'statistically significant' if p_val < 0.05 else 'weak'}** positive correlation (**r = {r:.2f}**) between amenity count and price. 
    
    **The "Amenity Lift":** Listings in the top 25% of amenities command **€{high_med:.0f}/night**—a **{price_lift:.0f}% increase** over the bottom 25% (**€{low_med:.0f}**). 
    
    Investing in more amenities is a proven lever for increasing nightly revenue.
    """)

st.markdown("---")

st.header("4. Key Drivers & Correlation Analysis")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Correlation Heatmap")
    
    # Define variables
    numeric_vars = ['price', 'bedrooms', 'amenities_count', 
                    'review_scores_rating', 'accommodates', 
                    'minimum_nights', 'host_total_listings_count']
    
    # Calculate correlation
    corr = filtered_df[numeric_vars].corr()
    
    # Create a mask for the upper triangle (Professional touch)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                center=0, annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": .8})
    
    ax.set_title('Numeric Variable Relationships', fontsize=10)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig)

with col_b:
    # --- Dynamic Calculations ---
    price_corr = corr['price'].drop('price').sort_values(ascending=False)
    strongest_var = price_corr.index[0]
    strongest_val = price_corr.iloc[0]
    
    # Check directions for dynamic text
    min_nights_val = corr.loc['minimum_nights', 'price']
    rating_val = corr.loc['review_scores_rating', 'price']
    
    st.caption("**Statistical Dependencies**")
    st.info(f"""
    **--- Dashboard Insight ---**
    **{strongest_var}** is the strongest price driver (**r = {strongest_val:.2f}**), confirming larger properties earn more.
    
    **Stay Length:** Minimum nights shows a **{'negative' if min_nights_val < 0 else 'positive'}** link (**{min_nights_val:.2f}**), suggesting pricing shifts for long-term stays.
    
    **Quality vs. Price:** Ratings have a **{abs(rating_val):.2f}** correlation, indicating price and quality are **{'weakly' if abs(rating_val) < 0.3 else 'moderately'}** linked.
    """)


# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit | Data: Airbnb Paris Listings")