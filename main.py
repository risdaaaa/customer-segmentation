import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from babel.numbers import format_currency
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.cm import get_cmap
from math import pi



# Cache for loading the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    datetime_columns = ["InvoiceDate"]
    df.sort_values(by="InvoiceDate", inplace=True)
    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

# Load the dataset
df = load_data()

# Function to filter data based on date range
@st.cache_data
def filter_data_by_date(df, start_date, end_date):
    return df[(df["InvoiceDate"] >= str(start_date)) & 
              (df["InvoiceDate"] <= str(end_date))]

# ========================================================
@st.cache_data
def calculate_rfm(df):
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (recent_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Sales', 'sum')
    ).reset_index()
    return rfm

#==========================================================================
# Streamlit sidebar for filtering by date range
min_date = df["InvoiceDate"].min()
max_date = df["InvoiceDate"].max()

with st.sidebar:
    st.image("https://raw.githubusercontent.com/risdaaaa/customer-segmentation/refs/heads/main/logo%20cluster.jpg", width=200)
    
    # Add a stylish title to the sidebar
    st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Customer Segmentation</h3>", unsafe_allow_html=True)
    
    # Add a description under the title
    st.markdown("<p style='text-align: center; color: #555;'>Select the date range for analysis and explore customer clusters.</p>", unsafe_allow_html=True)

    # Date range input with better styling
    start_date, end_date = st.date_input(
        label='Select Date Range', 
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
        key="date_range"
    )

    # Let user select the number of clusters
    n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=5, value=5)

    # Styling the sidebar section
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-color: #f4f4f9;
                padding: 10px;
                border-radius: 10px;
            }
            .sidebar h1 {
                font-size: 20px;
                font-weight: bold;
                color: #0073e6;
                text-align: center;
            }
            .social-icons {
                display: flex;
                justify-content: space-evenly;
                margin-top: 20px;
                padding: 10px;
                background-color: #eef2f3;
                border-radius: 10px;
                margin-top: 10px;
            }
            .social-icons img {
                transition: transform 0.3s ease;
            }
            .social-icons img:hover {
                transform: scale(1.2);
            }
        </style>
    """, unsafe_allow_html=True)

    # Displaying social icons with a nice hover effect and custom style
    st.markdown("""
        <h1>Connect with Me</h1>
        <div class="social-icons" style="padding: 10px; background-color: #eef2f3; border-radius: 10px; margin-top: 10px;">
            <a href="https://github.com/risdaaaa" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/github.png" width="50" height="50"/>
            </a>
            <a href="https://www.linkedin.com/in/dwi-krisdanarti/" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" width="50" height="50"/>
            </a>
            <a href="mailto:dwikrisda2@gmail.com" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/email.png" width="50" height="50"/>
            </a>
        </div>
    """, unsafe_allow_html=True)

# Filter the dataset based on the selected date range
main_df = filter_data_by_date(df, start_date, end_date)

#==============================
tab1, tab2 = st.tabs(["Customer and Product Analysis Dashboard", "Customer Segmentation Using K-Means Clustering"])
with tab1:
    #====================================================================
    st.title("Customer and Product Analysis Dashboard")
    #============================================================================
    # Add custom CSS for styling the grid and metrics
    st.markdown("""
        <style>
            .metric-title {
                font-size: 24px;
                font-weight: bold;
                color: #1e3a8a; /* Dark blue for the title */
            }
            .metric-value {
                font-size: 30px;
                font-weight: bold;
                color: #ff6f61; /* Soft red for the value */
            }
            .stMetrics {
                padding: 20px;
                background-color: #f3f4f6;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stColumn {
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Count the total number of customers (unique Customer IDs)
    total_customers = main_df['CustomerID'].nunique()

    # Count Unique Product IDs (e.g., 'StockCode' represents product IDs in your dataset)
    unique_product_ids = main_df['StockCode'].nunique()

    # Create a grid layout using Streamlit columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="stMetrics"><p class="metric-title">Total Number of Customers</p><p class="metric-value">{}</p></div>'.format(total_customers), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stMetrics"><p class="metric-title">Total Number of Products</p><p class="metric-value">{}</p></div>'.format(unique_product_ids), unsafe_allow_html=True)

    #==========================================================================
    ###### Product Analysis
    st.header("📊 Product Analysis")

    # Orders per customer
    orders_per_customer = main_df.groupby('CustomerID')['InvoiceNo'].nunique()

    # Distribution of orders per customer
    st.subheader("Distribution of Orders per Customer")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(orders_per_customer, bins=30, color='#66785F', edgecolor='black')
    ax.set_title('Distribution of Orders per Customer', fontsize=16)
    ax.set_xlabel('Number of Orders', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Top 5 and Bottom 5 Customers
    st.subheader("Top 5 and Bottom 5 Customers by Order Count")
    top_5_customers = orders_per_customer.sort_values(ascending=False).head(5)
    bottom_5_customers = orders_per_customer.sort_values(ascending=True).head(5)

    fig, axes = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
    colors = ["#66785F", "#B2C9AD", "#B2C9AD", "#B2C9AD", "#B2C9AD"]

    # Top 5 Customers
    sns.barplot(x=top_5_customers.index.astype(str), y=top_5_customers, palette=colors, ax=axes[0])
    axes[0].set_title('Top 5 Customers by Order Count', fontsize=15)

    # Bottom 5 Customers
    sns.barplot(x=bottom_5_customers.index.astype(str), y=bottom_5_customers, palette=colors, ax=axes[1])
    axes[1].invert_xaxis()
    axes[1].set_title('Bottom 5 Customers by Order Count', fontsize=15)

    st.pyplot(fig)

    # Best and Worst Performing Products
    st.subheader("Best and Worst Performing Products by Number of Sales")
    sum_order_items_df = main_df.groupby("Description")['Quantity'].sum().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

    # Top 5 Products
    sns.barplot(x="Quantity", y="Description", data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
    ax[0].set_title("Best Performing Products", fontsize=15)

    # Bottom 5 Products
    sns.barplot(x="Quantity", y="Description", data=sum_order_items_df.sort_values(by="Quantity", ascending=True).head(5), palette=colors, ax=ax[1])
    ax[1].invert_xaxis()
    ax[1].set_title("Worst Performing Products", fontsize=15)

    st.pyplot(fig)

    #===========================================================================================================================================================
    ##### Geographical Analysis
    st.header("🌍Geographical Analysis")
    st.subheader("Top and Bottom 5 Countries")
    # Menghitung jumlah pesanan berdasarkan negara
    country_order_counts = main_df['Country'].value_counts()

    # Mengambil 5 negara teratas dan terbawah
    top_5_countries = country_order_counts.head(5)
    bottom_5_countries = country_order_counts.tail(5)

    # Tentukan warna untuk grafik
    colors = ["#66785F", "#B2C9AD", "#B2C9AD", "#B2C9AD", "#B2C9AD"]

    # Membuat grid plot 1 baris dan 2 kolom
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

    # Plot 5 negara terbaik (Top 5)
    sns.barplot(x=top_5_countries.values, y=top_5_countries.index, palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Top 5 Countries by Order Count", loc="center", fontsize=15)
    ax[0].tick_params(axis='y', labelsize=12)

    # Plot 5 negara terburuk (Bottom 5)
    sns.barplot(x=bottom_5_countries.values, y=bottom_5_countries.index, palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].invert_xaxis()  # Membalikkan sumbu X
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Bottom 5 Countries by Order Count", loc="center", fontsize=15)
    ax[1].tick_params(axis='y', labelsize=12)

    # Menambahkan judul umum
    plt.suptitle("Top and Bottom 5 Countries by Order Count", fontsize=20)

    # Menampilkan plot
    st.pyplot(fig)

    #----------------------------------------------------
    # # Geolocation Analysis: Mapping countries with order counts
    # order_count = main_df['Country'].value_counts().reset_index()
    # order_count.columns = ['Country', 'OrderCount']

    # # Inisialisasi geolocator untuk mendapatkan koordinat negara
    # geolocator = Nominatim(user_agent="geoapi")

    # # Tambahkan koordinat (latitude, longitude) untuk setiap negara
    # def get_coordinates(country):
    #     try:
    #         location = geolocator.geocode(country)
    #         return location.latitude, location.longitude
    #     except:
    #         return None, None

    # order_count['Coordinates'] = order_count['Country'].apply(get_coordinates)
    # order_count[['Latitude', 'Longitude']] = pd.DataFrame(order_count['Coordinates'].tolist(), index=order_count.index)

    # # Hapus negara tanpa koordinat
    # order_count = order_count.dropna(subset=['Latitude', 'Longitude'])

    # # Inisialisasi peta
    # m = folium.Map(location=[20, 0], zoom_start=2)  # Lokasi awal peta global

    # # Tambahkan marker cluster
    # marker_cluster = MarkerCluster().add_to(m)

    # # Tambahkan marker berdasarkan lokasi
    # for _, row in order_count.iterrows():
    #     folium.Marker(
    #         location=[row['Latitude'], row['Longitude']],
    #         popup=f"Country: {row['Country']}<br>Order Count: {row['OrderCount']}",
    #         icon=folium.Icon(color="blue", icon="info-sign")
    #     ).add_to(marker_cluster)

    # # Display map in Streamlit
    # st.subheader("Order Count per Country Map")
    # st_folium(m, width=725)

    #====================================================
    ###### Customer Analysis
    st.header("👥Customer Analysis")

    # Add custom CSS for styling the grid and metrics
    st.markdown("""
        <style>
            .metric-title {
                font-size: 24px;
                font-weight: bold;
                color: #1e3a8a; /* Dark blue for the title */
            }
            .metric-value {
                font-size: 30px;
                font-weight: bold;
                color: #ff6f61; /* Soft red for the value */
            }
            .stMetrics {
                padding: 20px;
                background-color: #f3f4f6;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stColumn {
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Unique Customer IDs
    unique_customer_ids = main_df['CustomerID'].nunique()

    # Customer Activity Duration
    customer_activity = main_df.groupby('CustomerID').agg(
        FirstPurchase=('InvoiceDate', 'min'),
        LastPurchase=('InvoiceDate', 'max')
    ).reset_index()

    customer_activity['ActiveDuration'] = (customer_activity['LastPurchase'] - customer_activity['FirstPurchase']).dt.days

    # Calculate average active duration
    average_active_duration = customer_activity['ActiveDuration'].mean()

    # Create a grid layout using Streamlit columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="stMetrics"><p class="metric-title">Total Number of Customers</p><p class="metric-value">{}</p></div>'.format(unique_customer_ids), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stMetrics"><p class="metric-title">Average Active Duration</p><p class="metric-value">{:.2f} days</p></div>'.format(average_active_duration), unsafe_allow_html=True)

    #-----------------------------------------------------------
    # Distribution of Active Duration
    st.subheader("Customer Active Duration Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(customer_activity['ActiveDuration'], bins=30, kde=True, color='#66785F', edgecolor='black', ax=ax)
    ax.axvline(average_active_duration, color='red', linestyle='--', label=f'Average: {average_active_duration:.2f} days')
    ax.set_title('Customer Active Duration Distribution', fontsize=16)
    ax.set_xlabel('Active Duration (days)', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    #====================================================
    ###### Refund Analysis
    st.header("💸Returns Analysis")

    # Add custom CSS for styling the grid and metrics
    st.markdown("""
        <style>
            .metric-title {
                font-size: 24px;
                font-weight: bold;
                color: #1e3a8a; /* Dark blue for the title */
            }
            .metric-value {
                font-size: 30px;
                font-weight: bold;
                color: #ff6f61; /* Soft red for the value */
            }
            .stMetrics {
                padding: 20px;
                background-color: #f3f4f6;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stColumn {
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Identify refund orders
    returns_or_refunds = main_df[(main_df['Quantity'] < 0) | (main_df['InvoiceNo'].str.startswith('C'))]

    # Calculate percentage of orders with refunds
    total_orders = main_df['InvoiceNo'].nunique()
    refund_orders = returns_or_refunds['InvoiceNo'].nunique()
    refund_percentage = (refund_orders / total_orders) * 100

    # Count Unique Product IDs (e.g., 'StockCode' represents product IDs in your dataset)
    unique_product_ids = main_df['StockCode'].nunique()

    # Create a grid layout using Streamlit columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="stMetrics"><p class="metric-title">Refund Orders Percentage</p><p class="metric-value">{:.2f}%</p></div>'.format(refund_percentage), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stMetrics"><p class="metric-title">Total Number of Products</p><p class="metric-value">{}</p></div>'.format(unique_product_ids), unsafe_allow_html=True)

    #---------------------
    # Monthly Refunds
    refund = main_df[main_df['Quantity'] < 0].copy()
    refund.loc[:,'InvoiceDate'] = pd.to_datetime(refund['InvoiceDate'])
    refund.loc[:,'Month_Year'] = refund['InvoiceDate'].dt.strftime('%B %Y')
    monthly_refund = refund.groupby('Month_Year').size().reset_index(name='Refund_Count')
    monthly_refund = monthly_refund.sort_values('Month_Year')

    st.subheader("Monthly Refunds")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x='Month_Year', y='Refund_Count', data=monthly_refund, color="#66785F", edgecolor='black', ax=ax)
    ax.set_title('Refunds Count Over Time', fontsize=16)
    ax.set_xlabel('Month/Year', fontsize=12)
    ax.set_ylabel('Number of Refunds', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)


    #-------------------
    # Filter out categories with zero total orders to avoid division by zero
    category_orders = df.groupby('Description').size().reset_index(name='Total_Orders')
    category_orders = category_orders[category_orders['Total_Orders'] > 0]

    # Count returns for each category
    returns_or_refunds = df[df['Quantity'] < 0]
    category_returns = returns_or_refunds.groupby('Description').size().reset_index(name='Return_Count')

    # Merge with a minimum threshold of total orders
    category_data = pd.merge(category_orders, category_returns, on='Description', how='left').fillna(0)

    # Calculate return rate
    category_data['Return_Rate'] = category_data['Return_Count'] / category_data['Total_Orders']

    # Filter for categories with a significant number of total orders and varied return rates
    significant_categories = category_data[
        (category_data['Total_Orders'] > 10) &  # Ensure enough total orders
        (category_data['Return_Rate'] > 0) &    # Ensure some returns
        (category_data['Return_Rate'] < 1)      # Exclude 100% return rate
    ]

    # Sort and select top 5 with diverse return rates
    category_data_sorted = significant_categories.sort_values(by='Return_Rate', ascending=False)
    top_5_categories = category_data_sorted.head()

    # Print detailed information
    st.subheader("Top 5 Product Categories by Return Rate")
    # st.write(top_5_categories)

    # Set up the plot with 1 row, 2 columns (like your previous format)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Tentukan warna grafik
    colors = ["#66785F", "#B2C9AD", "#B2C9AD", "#B2C9AD", "#B2C9AD"]

    # Plot top 5 categories by return rate
    sns.barplot(x='Return_Rate', y='Description', data=top_5_categories, palette=colors, ax=ax)
    ax.set_title('Top 5 Product Categories by Return Rate', fontsize=16)
    ax.set_xlabel('Return Rate', fontsize=12)
    ax.set_ylabel('Product Category', fontsize=12)

    # Menambahkan label nilai di atas setiap batang
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=10)

    # Menampilkan plot
    plt.tight_layout()
    st.pyplot(fig)

    # Additional insights
    #st.subheader("Additional Insights:")
    #st.write(f"Total Unique Product Categories: {len(category_data)}")
    # st.write(f"Categories with Significant Orders and Returns: {len(significant_categories)}")

    #========================================
    ##### Sales, Cost, and Profit Analysis Analysis
    st.header("💰Sales, Cost, and Profit Analysis Analysis")

    # Menghitung Sales
    main_df['Sales'] = main_df['Quantity'] * main_df['UnitPrice']

    # Menghitung total sales
    total_sales = main_df['Sales'].sum()

    # Estimasi biaya dengan asumsi margin keuntungan 30%
    profit_margin = 0.30

    # Menghitung Cost
    main_df['Cost'] = main_df['Sales'] * (1 - profit_margin)

    # Menghitung total biaya dan total keuntungan
    total_cost = main_df['Cost'].sum()
    total_profit = total_sales - total_cost

    # Add custom CSS for styling the grid and metrics
    st.markdown("""
        <style>
            .metric-title {
                font-size: 24px;
                font-weight: bold;
                color: #1e3a8a; /* Dark blue for the title */
            }
            .metric-value {
                font-size: 30px;
                font-weight: bold;
                color: #ff6f61; /* Soft red for the value */
            }
            .stMetrics {
                padding: 20px;
                background-color: #f3f4f6;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stColumn {
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create a grid layout using Streamlit columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="stMetrics"><p class="metric-title">Total Cost</p><p class="metric-value">${:,.2f}</p></div>'.format(total_cost), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stMetrics"><p class="metric-title">Total Profit</p><p class="metric-value">${:,.2f}</p></div>'.format(total_profit), unsafe_allow_html=True)

    #------------------
    # Grouping by product and calculating metrics
    product_profit = main_df.groupby('Description').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Quantity=('Quantity', 'sum'),
        Total_Profit=('Sales', lambda x: x.sum() * profit_margin)
    ).reset_index()

    # Calculating profit margin
    product_profit['Profit_Margin'] = product_profit['Total_Profit'] / product_profit['Total_Sales']

    # Filtering significant products
    significant_products = product_profit[
        (product_profit['Total_Sales'] > 0) &  # Excluding products with zero sales
        (product_profit['Total_Quantity'] > 10) &  # Minimum of 10 units sold
        (product_profit['Profit_Margin'] > 0) &  # Positive profit margin
        (product_profit['Profit_Margin'] < 1)  # Excluding 100% profit margin
    ]

    # Sorting and selecting the top 5 products with diverse profit margins
    top_products = significant_products.sort_values(by='Profit_Margin', ascending=False).head(5)

    # Displaying detailed information
    st.subheader("Top 5 Products by Profit Margin")
    st.write(top_products)

    # Wawasan tambahan
    # st.subheader("Wawasan Tambahan:")
    # st.write(f"Total Produk Unik: {len(product_profit)}")
    # st.write(f"Produk dengan Penjualan dan Margin Signifikan: {len(significant_products)}")

    #==================================
    # Customer Satisfaction & Dissatisfaction analysis
    st.header("😊Customer Satisfaction & Dissatisfaction Analysis")

    # Create Sales column
    main_df['Sales'] = main_df['Quantity'] * main_df['UnitPrice']

    # Satisfaction Rate (proxy: high purchase quantities, threshold = 50)
    satisfaction_threshold = 0  # Quantity threshold for high satisfaction
    satisfaction_data = main_df[main_df['Quantity'] >= satisfaction_threshold]
    product_satisfaction = satisfaction_data.groupby('Description').agg(
        Total_Quantity=('Quantity', 'sum'),
        Total_Sales=('Sales', 'sum')
    ).reset_index()
    product_satisfaction['Avg_Satisfaction_Score'] = np.clip(
        product_satisfaction['Total_Quantity'] / product_satisfaction['Total_Quantity'].max() * 5, 0, 5)

    # Dissatisfaction Rate (proxy: refund/return rates, negative quantities)
    dissatisfaction_data = main_df[main_df['Quantity'] < 0]
    product_dissatisfaction = dissatisfaction_data.groupby('Description').agg(
        Total_Returns=('Quantity', 'sum'),
        Total_Refunds=('Sales', 'sum')
    ).reset_index()
    product_dissatisfaction['Avg_Dissatisfaction_Score'] = np.clip(
        abs(product_dissatisfaction['Total_Returns']) / abs(product_dissatisfaction['Total_Returns']).max() * 5, 0, 5)

    # Merge satisfaction and dissatisfaction data
    product_feedback = pd.merge(
        product_satisfaction[['Description', 'Avg_Satisfaction_Score']],
        product_dissatisfaction[['Description', 'Avg_Dissatisfaction_Score']],
        on='Description', how='outer'
    ).fillna(0)  # Fill missing scores with 0

    # Sort by dissatisfaction score and filter top 5 products
    top_feedback_products = product_feedback.nlargest(5, 'Avg_Dissatisfaction_Score')

    # Step 2: Create a stacked bar chart for satisfaction and dissatisfaction using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bars
    bars_satisfaction = ax.barh(top_feedback_products['Description'], top_feedback_products['Avg_Satisfaction_Score'], color='#66785F', label='Satisfaction')
    bars_dissatisfaction = ax.barh(top_feedback_products['Description'], top_feedback_products['Avg_Dissatisfaction_Score'], left=top_feedback_products['Avg_Satisfaction_Score'], color='#B2C9AD', label='Dissatisfaction')

    # Add labels for each bar
    for bar in bars_satisfaction:
        ax.text(bar.get_width() - bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}', ha='center', va='center', color='white')
    for bar in bars_dissatisfaction:
        ax.text(bar.get_width() - bar.get_width() / 2 + top_feedback_products['Avg_Satisfaction_Score'].values[0], bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}', ha='center', va='center', color='white')

    # Add labels and title
    ax.set_xlabel('Score (Out of 5)')
    ax.set_title('Satisfaction vs Dissatisfaction by Product (Top 5)')
    ax.legend()

    # Display the plot
    plt.tight_layout()
    st.pyplot(fig)

    #=================================
    # Streamlit header for RFM metrics
    st.header("💡RFM Analysis")

    # Mendefinisikan recent_date sebagai tanggal terbaru dalam dataset
    recent_date = main_df["InvoiceDate"].max()
    # Calculate RFM metrics
    rfm = calculate_rfm(main_df)

    st.subheader("Best Customer Based on RFM Parameters")

    # Calculate the average RFM metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_recency = round(rfm['Recency'].mean(), 1)
        st.metric("Average Recency (days)", value=avg_recency)

    with col2:
        avg_frequency = round(rfm['Frequency'].mean(), 2)
        st.metric("Average Frequency", value=avg_frequency)

    with col3:
        avg_monetary = format_currency(rfm['Monetary'].mean(), "USD", locale='en_US')  # Ganti "USD" jika menggunakan mata uang lain
        st.metric("Average Monetary", value=avg_monetary)

    # Create a shortened version of customer_id for plotting (optional, jika tidak ada, bisa diabaikan)
    rfm['short_customer_id'] = rfm['CustomerID'].astype(str).str[:8]  # Ambil 8 karakter pertama

    # Set up plots for Recency, Frequency, and Monetary
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))  # Adjust figure size if needed

    # Define a consistent color palette
    colors = ['#66785F'] * 5  # Golden color for uniformity

    # Plot for Recency
    sns.barplot(
        y="Recency", 
        x="short_customer_id", 
        data=rfm.sort_values(by="Recency", ascending=True).head(5), 
        palette=colors, 
        ax=ax[0]
    )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("By Recency (days)", loc="center", fontsize=20)
    ax[0].tick_params(axis='x', labelsize=14)

    # Plot for Frequency
    sns.barplot(
        y="Frequency", 
        x="short_customer_id", 
        data=rfm.sort_values(by="Frequency", ascending=False).head(5), 
        palette=colors, 
        ax=ax[1]
    )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("By Frequency", loc="center", fontsize=20)
    ax[1].tick_params(axis='x', labelsize=14)

    # Plot for Monetary
    sns.barplot(
        y="Monetary", 
        x="short_customer_id", 
        data=rfm.sort_values(by="Monetary", ascending=False).head(5), 
        palette=colors, 
        ax=ax[2]
    )
    ax[2].set_ylabel(None)
    ax[2].set_xlabel(None)
    ax[2].set_title("By Monetary", loc="center", fontsize=20)
    ax[2].tick_params(axis='x', labelsize=14)

    # Rotate x-axis labels for better readability
    for axis in ax:
        axis.tick_params(axis='x', rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)

# Content for "Customer Segmentation Using K-Means Clustering"
with tab2:
    st.title("Customer Segmentation Using K-Means Clustering")
    
    ##### find number of cluster
    # Elbow Method for finding optimal k
    inertia = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=123)
        kmeans.fit(rfm)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Method
    st.subheader('Elbow Method for Optimal k')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_values, inertia, marker='o', linestyle='-', color='darkorange', markersize=8, linewidth=2)
    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Silhouette Score Calculation
    arr_silhouette_score_euclidean = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=123).fit(rfm)
        preds = kmeans.predict(rfm)
        score_euclidean = silhouette_score(rfm, preds, metric='euclidean')
        arr_silhouette_score_euclidean.append(score_euclidean)

    # Plot Silhouette Score vs Number of Clusters
    st.subheader('Silhouette Score vs Number of Clusters')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=range(2, 11), y=arr_silhouette_score_euclidean, color='#0066CC', linewidth=4, ax=ax)
    sns.scatterplot(x=range(2, 11), y=arr_silhouette_score_euclidean, s=150, color='#FF5733', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Silhouette Score vs Number of Clusters', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    #============================================
    ##### Clustering Section

    @st.cache_data
    def perform_clustering(rfm, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=123)
        kmeans.fit(rfm[['Recency', 'Frequency', 'Monetary']])
        rfm['cluster'] = kmeans.labels_
        return rfm

    # Perform K-Means Clustering
    rfm = perform_clustering(rfm, n_clusters)

    # Mengelompokkan data dan menghitung nilai rata-rata per cluster
    cluster_summary = rfm.groupby('cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(1)

    # Menampilkan Summary
    st.subheader(f"Customer Segments (n={n_clusters})")
    st.write(cluster_summary)

    # Visualisasi K-Means Clusters
    palette = sns.color_palette("Spectral", n_colors=n_clusters)

    # Scatter plot untuk memvisualisasikan kluster
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=rfm,
        x='Recency',
        y='Monetary',
        hue='cluster',
        size='Frequency',
        sizes=(20, 200),  # Skala ukuran titik
        palette=palette,
        edgecolor='black',
        alpha=0.9,
        ax=ax
    )

    # Menambahkan elemen plot
    ax.set_title('Customer Segmentation Using K-Means Clustering', fontsize=16, color='darkblue')
    ax.set_xlabel('Recency (Days Since Last Purchase)', fontsize=12, color='darkgreen')
    ax.set_ylabel('Monetary (Total Spending)', fontsize=12, color='darkgreen')
    ax.legend(
        title='Customer Cluster',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10
    )
    ax.grid(True, linestyle='--', alpha=0.6, color='gray')

    # Menampilkan plot
    plt.tight_layout()
    st.pyplot(fig)

    # Count of customers per cluster
    cluster_count = rfm['cluster'].value_counts().reset_index()
    cluster_count.columns = ['cluster', 'count']
    cluster_count['percentage (%)'] = round((cluster_count['count']/len(rfm))*100, 2)
    cluster_count = cluster_count.sort_values(by=['cluster']).reset_index(drop=True)

    st.subheader("Cluster Count and Percentage")
    st.write(cluster_count)

    # Melting data for visualization
    df_melt = pd.melt(cluster_summary.reset_index(), id_vars='cluster', value_vars=['Recency', 'Frequency', 'Monetary'],
                    var_name='Metric', value_name='Value')

    # Visualizing Patterns of Clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.pointplot(data=df_melt, x='Metric', y='Value', hue='cluster', palette='Spectral', markers="o", scale=1.5, ax=ax)

    # Adding labels and title
    ax.set_title('Customer Pattern by RFM Clusters', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Metric', fontsize=14, labelpad=10)
    ax.set_ylabel('Value', fontsize=14, labelpad=10)

    # Adding grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding legend
    ax.legend(title='Cluster', title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

    # Tight layout to avoid cut-off
    plt.tight_layout()
    st.pyplot(fig)
    #=======================================================
    # Tambahkan radar chart untuk RFM Clusters
    def radar_chart_rfm(rfm, labels, n_clusters):
        categories = ['Recency', 'Frequency', 'Monetary']
        N = len(categories)

        # Hitung nilai rata-rata setiap cluster
        cluster_means = rfm.groupby(labels).mean()[categories]
        
        # Tambahkan nilai awal untuk radar chart
        values = cluster_means.values.tolist()
        for i in range(len(values)):
            values[i] += values[i][:1]

        # Hitung sudut untuk radar chart
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Buat plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

        # Tambahkan data untuk setiap cluster
        cmap = get_cmap("tab10")
        for i in range(n_clusters):
            ax.plot(angles, values[i], linewidth=2, linestyle='solid', label=f'Cluster {i+1}', color=cmap(i))
            ax.fill(angles, values[i], alpha=0.25, color=cmap(i))

        # Tambahkan label kategori
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Tambahkan legenda
        plt.title('Radar Chart of RFM Clusters', size=20, color='black', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        return fig

    # ========================================================
    # Tambahkan proses clustering dan radar chart ke Streamlit
    recent_date = main_df["InvoiceDate"].max()
    rfm = calculate_rfm(main_df)

    # Lakukan scaling data RFM (opsional)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    rfm_scaled = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_scaled[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm_scaled)

    # Jalankan algoritma K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Radar chart berdasarkan hasil clustering
    st.subheader("Radar Chart RFM Clusters")
    fig = radar_chart_rfm(rfm, rfm['Cluster'], n_clusters)
    st.pyplot(fig)

        
    # Function to add background color to each cluster
    def add_background(color):
        st.markdown(f"""
        <style>
        .cluster-box {{
            background-color: {color};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            color: white;
        }}
        </style>
        """, unsafe_allow_html=True)

    if n_clusters == 5:
        st.write("### 🚀 Here is the analysis and recommendations for each cluster:")

        # Cluster 0 - Dormant Customers
        add_background("#f28d30")  # Orange background
        st.markdown("<div class='cluster-box'>", unsafe_allow_html=True)
        st.write("#### 1. Cluster 0 - \"Dormant Customers\" 😴")
        st.write("**Characteristics:**")
        st.write("Customers in this cluster rarely make purchases (low frequency), have low transaction values (low monetary), and have not shopped for a long time (high recency).")
        st.write("**Recommendations:**")
        st.write("- 🎁 Offer special promotions like discounts or coupons to re-engage these customers.")
        st.write("- 📧 Send reminder emails or exclusive offers to reactivate them.")
        st.write("- 🔄 Launch retention campaigns focusing on rebuilding engagement.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Cluster 1 - Engaged Shoppers
        add_background("#3c8dbc")  # Blue background
        st.markdown("<div class='cluster-box'>", unsafe_allow_html=True)
        st.write("#### 2. Cluster 1 - \"Engaged Shoppers\" 🛍️")
        st.write("**Characteristics:**")
        st.write("These customers are moderately active with a medium purchase frequency and decent transaction value. They haven't shopped too long ago (low recency).")
        st.write("**Recommendations:**")
        st.write("- 🎯 Implement loyalty programs to keep them active, like reward points or exclusive offers.")
        st.write("- 🔄 Run upselling and cross-selling campaigns to increase transaction value.")
        st.write("- 🤝 Maintain good relationships with proactive customer service.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Cluster 2 - Loyal High-Value Customers
        add_background("#27ae60")  # Green background
        st.markdown("<div class='cluster-box'>", unsafe_allow_html=True)
        st.write("#### 3. Cluster 2 - \"Loyal High-Value Customers\" 🏆")
        st.write("**Characteristics:**")
        st.write("These are your best customers. They are highly active (high frequency), have large transaction values (high monetary), and have recently made a purchase (very low recency).")
        st.write("**Recommendations:**")
        st.write("- 🎁 Reward their loyalty by offering exclusive access to new products or premium services.")
        st.write("- 👑 Treat them as VIP customers with additional benefits to maintain long-term relationships.")
        st.write("- 🌟 Encourage them to become brand ambassadors or provide testimonials.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Cluster 3 - At-Risk High-Value Customers
        add_background("#e74c3c")  # Red background
        st.markdown("<div class='cluster-box'>", unsafe_allow_html=True)
        st.write("#### 4. Cluster 3 - \"At-Risk High-Value Customers\" ⚠️")
        st.write("**Characteristics:**")
        st.write("These customers have a medium purchase frequency, decent transaction value, but their last purchase was some time ago (medium recency).")
        st.write("**Recommendations:**")
        st.write("- 🔄 Offer incentives to encourage them to shop again, such as referral programs or time-limited discounts.")
        st.write("- 🛒 Send personalized communications to attract their attention, such as product recommendations based on their purchase history.")
        st.write("- 🎁 Offer bundling packages to increase transaction value.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Cluster 4 - Super VIPs
        add_background("#8e44ad")  # Purple background
        st.markdown("<div class='cluster-box'>", unsafe_allow_html=True)
        st.write("#### 5. Cluster 4 - \"Super VIPs\" 🌟")
        st.write("**Characteristics:**")
        st.write("These are your super top-tier customers. They shop very frequently (very high frequency), have very large transaction values (very high monetary), and have made a recent purchase (very low recency).")
        st.write("**Recommendations:**")
        st.write("- 🏆 Offer special rewards like personalized services or exclusive discounts for bulk purchases.")
        st.write("- 🎉 Involve them in exclusive events like product launches or special events.")
        st.write("- 💬 Consider co-creation strategies, such as asking for their input on product or service development.")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.write("### ❌ The number of clusters is not optimal. Try selecting a higher number of clusters or adjust based on your analysis.")
        st.write("📊 The right number of clusters will help you gain better insights for your business or analysis.")

    
