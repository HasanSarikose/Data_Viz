📊 Shopping Behaviour Intelligence Dashboard
CEN445 – Introduction to Data Visualization (2025–2026)

Interactive analytics system built with Python + Streamlit, featuring multivariate visual exploration, advanced visualizations, geospatial analytics, and machine learning–powered insights.

🗂️ Project Overview

This dashboard analyzes the Shopping Behaviour dataset, containing demographic and behavioral details such as:

    Customer age

    Purchase amount

    Product categories

    Payment method

    Review ratings

    Subscription status

    Seasonal purchase trends

The project focuses on turning raw consumer data into interactive, visual, and ML-supported insights.

📁 Dataset

    Source: [Kaggle — Shopping Behaviour Dataset](https://www.kaggle.com/datasets/nalisha/shopping-behaviour-and-product-ranking-dateset?resource=download)

    Rows: 3,900+

    Columns: 17

    Types: Numerical, categorical, hierarchical, and temporal features

    Cleaning:

    All incomplete rows automatically removed

    Columns normalized

    State names auto-mapped to US state codes (for the map)

🧼 Data Preprocessing

The application includes a dedicated data-cleaning pipeline:

* Automatic handling of missing values
* Removal of incomplete rows
* On-the-fly dataset upload & replacement
* State-column detection for maps
* Smart row sampling for performance
* Shared global filters applied across all charts

**🖥️ How to Run the Dashboard**
*     `pip install -r requirements.txt
*     streamlit run main.py`

Python 3.9+ recommended.

**🎨 Dashboard Features**

The dashboard includes 11 fully interactive visualizations (requirement: 9), with 8 advanced techniques (requirement: 6).
All charts support hover tooltips, brushing, zooming, sliders, dropdowns, and automated insight generation.

📊 Visualization List (11 Total)

**Önder Hüsnü Kabadayı (3 Visualizations – Exploratory/Statistical)**
1. **Parallel Coordinates:** Shows how Age, Purchase Amount, Review Rating, and Previous Purchases interact.
Brushing reveals high-spending demographic slices and gender-based differences.

2. **Histogram + KDE:** Explores the distribution of key metrics (Age, Purchase Amount, Review Rating):
Highlights skewness, outliers, and peaks.

3. **Season Line + Slider:** Seasonal purchase behavior from Winter → Autumn with an age-range slider.
Allows age-specific comparison of seasonal spending trends.

**Hasan Sarıköse (5 Visualizations – Advanced & ML)**
**1. Payment → Category → Subscription Sankey**
    
    Advanced multi-level flow analysis with:
    
    Custom metrics (count, purchase amount, previous purchases)
    
    Flow filtering, hover paths, drag interactivity
    
    Automated insights (dominant payment method, strongest path)

**2. Sunburst / Treemap**

Hierarchical navigation of Category → Item Purchased → Color with sum/mean/count aggregation.
Includes insight extraction (dominant categories, item diversity).

**3. Payment ↔ Category Network Diagram**

Bipartite network visualizing co-occurrence strength:

Edge thickness = purchase frequency

Node layers (payment left, category right)

Insight engine: strongest link, most connected payment method

**4. USA Category Map (3 Modes)**

A complete geographic analytics module:

Mode A: Most popular category per state

Mode B: Metric choropleth (count/amount)

Mode C: Choropleth + category breakdown on hover

Supports automatic state detection and state-code mapping.

**5. K-Means Customer Segmentation (Machine Learning)**

Interactive ML module with:

Feature selection

Adjustable cluster count (k)

Scaled 2D visualization

Insights: largest cluster, purchase power, behavioral segments

**Mustafa Şekeroğlu (3 Visualizations – Distributions & Multidimensional Data)**

**1. Violin Plot**

    Visualizes the distribution of Purchase Amount and Review Rating across:
    
    Age groups
    
    Customer segments
    Includes boxplot and jittered points for detail.

**2. 3D Scatter Plot**

    Examines relationships between:
    
    Age
    
    Purchase Amount
    
    Previous Purchases
    With color encoding for categorical groups.

**3. Treemap**

    Shows categorical distribution (segments or rating levels) with size & color based on aggregated values.
    Ideal for summarizing hierarchical or categorical compositions.

**🧠 Machine Learning Component
K-Means Clustering**

* Feature selection (Age, Purchase Amount, Review Rating, Previous Purchases, etc.)


* Automatic scaling


* 2D visualization with cluster coloring


* Insight extraction for business interpretation


* (Optional ML requirement completed)

**💬 Insight Engine – Automated Commentary**

* Every chart includes an automated insight generator that detects:


* Dominant categories


* Behavioral clusters


* Seasonal peaks


* Strongest Sankey flows


* Top-performing states


* Outlier-heavy distributions


* High-spending customer groups


* This transforms visualizations into narrative analytics, directly satisfying the “insight generation” component of the assignment.

**🧱 Project Architecture**
 `   /charts
        /onder
        /hasan
        /mustafa
        /ml
    /utils
    main.py
    README.md
    shopping_behavior.csv`


* Modular chart structure per contributor


* Shared global filter pipeline


* Singleton dataset management


* Animated CSS styling + responsive grid layout

**👥 Team Contributions Summary**

Hasan Sarıköse

1. Repository creation & full architecture design

3. Sankey diagram

5. Sunburst / Treemap

7. Payment-Category Network

9. USA 3-Mode Category Map

11. K-Means Clustering (ML module)

13. Automated insight engine

15. Data cleaning + preprocessing pipeline

17. Interactivity & UX design

19. Dataset upload & cleaning system

Önder Hüsnü Kabadayı

1. Parallel Coordinates

3. Histogram + KDE

5. Seasonal Trend + Slider

7. Distribution analysis, demographic slicing

9. Shared filter integration



Mustafa Şekeroğlu

1. Violin Plot

3. 3D Scatter

5. Treemap

7. Category distribution analysis

9. Aesthetic tuning & plot adjustments