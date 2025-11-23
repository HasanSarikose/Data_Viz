# 📊 Shopping Behaviour Intelligence Dashboard
CEN445 – Introduction to Data Visualization (2025–2026)

Interactive analytics system built with Python + Streamlit, featuring multivariate visual exploration, advanced visualizations, geospatial analytics, and machine learning–powered insights.

### **🗂️ Project Overview**

This dashboard analyzes the Shopping Behaviour dataset, containing demographic and behavioral details such as:

* Customer age

* Purchase amount

* Product categories

* Payment method

* Review ratings

* Subscription status

* Seasonal purchase trends

The project focuses on turning raw consumer data into interactive, visual, and ML-supported insights.

### **📁 Dataset**

* Source: [Kaggle — Shopping Behaviour Dataset](https://www.kaggle.com/datasets/nalisha/shopping-behaviour-and-product-ranking-dateset?resource=download)

* Rows: 3,900+

* Columns: 17

* Types: Numerical, categorical, hierarchical, and temporal features

* Cleaning:

* All incomplete rows automatically removed

* Columns normalized

* State names auto-mapped to US state codes (for the map)

### **🧼 Data Preprocessing**

The application includes a dedicated data-cleaning pipeline:

* Automatic handling of missing values
* Removal of incomplete rows
* On-the-fly dataset upload & replacement
* State-column detection for maps
* Smart row sampling for performance
* Shared global filters applied across all charts

### **🖥️ How to Run the Dashboard**
*     pip install -r requirements.txt
*     streamlit run main.py
Python 3.9+ recommended.

### **🎨 Dashboard Features**

The dashboard includes 11 fully interactive visualizations (requirement: 9), with 8 advanced techniques (requirement: 6).
All charts support hover tooltips, brushing, zooming, sliders, dropdowns, and automated insight generation.

# **📊 Visualization List (11 Total)**

## **Hasan Sarıköse (5 Visualizations – Advanced & ML)**

### **1. Payment → Category → Subscription Sankey**
- Visualizes multi-level payment → category → subscription funnels.
- Supports custom metrics (count, purchase amount, previous purchases).
- Hover paths, drag interactions, and flow filtering keep the diagram readable.
- Automated commentary calls out dominant payment methods and strongest paths.
- Highlights leakage by exposing thin links between steps.

### **2. Sunburst**
- Explores Category → Item Purchased → Color with sum/mean/count aggregations.
- Lets viewers collapse/expand levels to understand category depth.
- Reveals dominant items or colors inside each category.
- Works as both an overview and drill-down hierarchy explorer.

### **3. Payment ↔ Category Network Diagram**
- Bipartite layout shows co-occurrence strength between payment methods and categories.
- Edge thickness reflects purchase frequency, spotlighting strong pairs.
- Node layering separates payment nodes (left) and categories (right) for clarity.
- Automated insight engine identifies the most connected payment type.

### **4. USA Category Map (3 Modes)**
- Provides a full geographic module for category performance.
- Mode A: Most popular category per state (categorical fills).
- Mode B: Choropleth by metric (count or purchase amount).
- Mode C: Choropleth plus category breakdown on hover.
- Auto-detects state names/codes and supports map zoom/hover storytelling.

### **5. Machine Learning – K-Means Clustering**
- Clusters shoppers using Age, Purchase Amount, Review Rating, Previous Purchases, etc.
- Handles automatic scaling and parameter selection for stability.
- Visualizes clusters in 2D with color-coded centroids.
- Generates business-friendly interpretation notes for each persona.

**6. Insight Engine – Automated Commentary**
* Every chart includes an automated insight generator that detects:
* Dominant categories
* Behavioral clusters
* Seasonal peaks
* Strongest Sankey flows
* Top-performing states
* Outlier-heavy distributions
* High-spending customer groups
* This transforms visualizations into narrative analytics, directly satisfying the “insight generation” component of the assignment.




## **Hüsnü Önder Kabadayı (3 Visualizations – Exploratory/Statistical)**
### **1. Parallel Coordinates**
- Visualizes multiple variables simultaneously.
- Shows relationships among Age, Purchase Amount, Review Rating, and Previous Purchases.
- Brushing allows filtering specific value ranges for deeper analysis.
- Highlights high-spending demographic segments.
- Reveals potential gender-based behavioral differences.
- Makes correlation patterns and upward/downward trends more visible.
### **. Insight**
- Brushing Age and Purchase Amount instantly exposes dense bundles where loyal, high-ticket cohorts concentrate.
- The color channel mirrors the selected KPI, so sudden hue shifts flag Review Rating or Previous Purchase anomalies without leaving the chart.

### **2. Histogram + KDE**
- Displays the distribution of key metrics (Age, Purchase Amount, Review Rating).
- KDE smoothens the density curve for clearer interpretation.
- Outliers can be identified quickly.
- Highlights skewness and symmetry/asymmetry in the data.
- Shows peaks where user behavior is most concentrated.
- Enables comparison of distributions across different groups (e.g., gender).
### **. Insight**
- Long tails expose which categories generate extreme spend or ratings, letting us call out risky or high-opportunity cohorts.
- The KDE overlay smooths noisy histogram bars, so multi-modal behavior pops when comparing gender or season filters.

### **3. Season Line + Slider**
- Illustrates seasonal purchasing behavior from Winter → Autumn.
- Shows clear upward or downward spending trends over time.
- Age-range slider enables filtering by specific age groups.
- Distinguishes seasonal patterns for young, middle-aged, and older users.
- Helps compare how seasonal changes influence spending habits.
- Provides an interactive, user-controlled analysis experience.
### **. Insight**
- Locking the age slider makes it easy to narrate exactly when each cohort peaks from Winter → Autumn.
- The range slider zoom highlights micro-season spikes (e.g., holidays), so contrasting age groups remains interactive.


## **Mustafa Şekeroğlu (3 Visualizations – Distributions & Multidimensional Data)**

### **1. Violin Plot**
- Displays Purchase Amount and Review Rating distributions across age groups and customer segments.
- Combines violin, box, and jitter marks for both density and individual outlier context.
- Highlights which cohorts have wide variance versus tight spending behavior.
- Useful for spotting retention risks or premium buyer pools.

### **2. 3D Scatter Plot**
- Plots Age, Purchase Amount, and Previous Purchases simultaneously.
- Color encodes categorical groups to show how segments distribute in 3D space.
- Helps detect natural clusters (e.g., young heavy spenders) worth targeted campaigns.
- Adjustable markers and camera angles aid interactive storytelling.

### **3. Treemap**
- Summarizes category or rating hierarchies with area (value) plus color (performance).
- Quickly reveals dominant segments and their proportional contribution.
- Nested layout clarifies how sub-groups roll up into parent categories.
- Ideal for comparing categorical mixes or highlighting revenue concentration.


## 🧱 Project Architecture

```plaintext
project-root/
│
├── main.py
├── README.md
├── requirements.txt
│
├── data/
│   └── shopping_behavior.csv
│
├── utils/
│   ├── data_loader.py
│   └── __init__.py
│
├── charts/
│   ├── hasan/
│   │   ├── hasan_charts.py
│   │   ├── ml_charts.py
│   │   └── __init__.py
│   │
│   ├── onder/
│   │   ├── onder_charts.py
│   │   └── __init__.py
│   │
│   ├── mustafa/
│   │   ├── mustafa_charts.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
└── __init__.py
```



* Modular chart structure per contributor


* Shared global filter pipeline


* Singleton dataset management


* Animated CSS styling + responsive grid layout

## **👥 Team Contributions Summary**

**Hasan Sarıköse**

1. Repository creation & full architecture design

3. Sankey diagram

5. Sunburst

7. Payment-Category Network

9. USA 3-Mode Category Map

11. K-Means Clustering (ML module)

13. Automated insight engine

15. Data cleaning + preprocessing pipeline

17. Creating UX design

19. Dataset upload & cleaning system

**Hüsnü Önder Kabadayı**

1. Parallel Coordinates

2. Histogram + KDE

3. Seasonal Trend + Slider

4. Distribution analysis, demographic slicing

5. Shared filter integration
6. Interactivity & UX design
7. Aesthetic Tuning & Plot Adjustments



**Mustafa Şekeroğlu**

1. Violin Plot

2. 3D Scatter Plot

3. Treemap

4. Category distribution analysis

. Aesthetic tuning & plot adjustments
