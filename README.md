
## 🛠️ Technical Stack

- **Language**: Python 3.12
- **Web Framework**: Streamlit
- **Data Libraries**: Pandas, NumPy, Matplotlib, Seaborn, SciPy
- **Environment**: Ubuntu (Linux)

---

## 🧹 Data Cleaning & Engineering
you can download the dataset using this link : https://www.kaggle.com/datasets/abaghyangor/airbnb-paris/data

### Outlier Management Strategy
- **Price (Continuous)**: IQR method (Q3 + 1.5×IQR) for statistical filtering
- **Bedrooms (Discrete)**: 99th percentile 

### Feature Engineering
- **amenities_count**: Extracted from string-represented lists to enable price correlation analysis

### Data Integrity
- **Visual Honesty**: Missing categorical metrics labeled explicitly ("Missing", "No Recent Activity")
- **Host Consistency**: Nulls filled using host's other active listings when available
- **Neighbourhood Imputation**: Review scores filled using arrondissement median

### Analytical Scope
- Univariate: Price, bedrooms, neighbourhood supply distributions
- Multivariate: Price vs neighbourhood, amenities vs price, room type analysis
- Correlations: Full numeric variable correlation matrix

---


## 🚀 Streamlit Dashboard

An interactive web dashboard visualizing Airbnb Paris listings with comprehensive EDA-driven insights.

### Features

#### 📊 Interactive Filters
- Price range slider (€)
- Room type multi-select
- Neighbourhood multi-select with smart defaults

#### 📈 Visualizations (from EDA)
- **Price Distribution** (linear + log scale) with mean/median indicators
- **Room Type Distribution** (donut chart) with percentages
- **Top 20 Neighbourhoods** (horizontal bar) with listing counts
- **Price by Neighbourhood** (boxplot) — top 15 by median price
- **Amenities vs Price** (scatter + trend line) colored by bedrooms
- **Correlation Matrix** (heatmap) — numeric variables relationships
- **Dataset Overview** — total listings, avg/median prices
- **Sample Data Table** — browse filtered listings

### Running the Dashboard

```bash
cd streamlit/
streamlit run dashboard.py
```

Open browser to `http://localhost:8501`

### Deployment

Deploy to Streamlit Cloud:
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and deploy

---

# 📍 Airbnb Paris – Data Analysis & Dashboard

Airbnb Listings and Reviews in Paris – Analysis of Trends, Pricing, and Market Insights

### Dataset Source
Download dataset: [Kaggle - Airbnb Paris](https://www.kaggle.com/datasets/abaghyangor/airbnb-paris/data)

---

## 📁 Project Structure

```
airbnb-paris/
├── README.md                              # Project documentation
├── data/
│   ├── raw/                               # Original downloaded data
│   │   ├── Listings.csv                   # Raw listings data
│   │   └── Listings_data_dictionary.csv   # Data dictionary
│   └── cleaned/                           # Processed data (parquet)
├── notebooks/
│   ├── EDA.ipynb                          # Exploratory Data Analysis
│   └── eda-listing.ipynb                  # Additional EDA notebook
└── streamlit/
    ├── dashboard.py                       # Main dashboard application
    └── paris_airbnb_cleaned.parquet       # Cleaned dataset (cached)
```

---


## 📊 Key Insights

- **Top Markets**: Top 5 neighbourhoods account for ~40% of all listings
- **Price Variance**: Median price varies 2-3x across neighbourhoods
- **Amenities Impact**: Properties with high amenity counts command 20-30% price premium
- **Room Type Distribution**: Entire homes dominate the Paris market (~60%)

## 🤝 Contributing

Feel free to open issues or PRs for improvements to the analysis or dashboard.

## 📝 License

This project is for educational and analytical purposes.