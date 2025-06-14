# 🛍️ User Behavior Analysis – EDA & Insights

Welcome to the **User Detection & Behavior Analysis** project! This initiative delves into the intricacies of online shopping behaviors, aiming to uncover patterns and insights that can enhance user experience and drive business decisions.

![Duration vs Purchases](duration_vs_purchases.png)

## 📌 Project Objectives

- Simulate a user behavior dataset that mimics real-world e-commerce platform activity.
- Clean and preprocess the data, including handling missing values, outliers, and scaling.
- Perform detailed **Exploratory Data Analysis (EDA)** using `pandas`, `matplotlib`, and `seaborn`.
- Generate static and interactive visualizations to explore:
  - Session behavior
  - Purchase frequency
  - Bounce rate differences
  - Visitor types (New vs Returning)
  - Region-specific purchase rates
- Export a structured `.csv` dataset and summary `.json` chart config for future use in dashboards.

---

## 🧾 Dataset Overview

If no external dataset is found, a synthetic dataset of **500 users** is generated using `numpy` and `uuid`.  
It includes the following features:

| Feature                 | Description                                               |
|------------------------|-----------------------------------------------------------|
| `user_id`              | Unique identifier for each user                           |
| `age`                  | Age of the visitor                                        |
| `session_duration`     | Total time spent on the website (minutes)                 |
| `page_views`           | Number of pages viewed during the session                 |
| `purchases`            | Number of purchases made during session                   |
| `time_on_product_page` | Time spent on product pages                               |
| `device`               | Device used (Mobile, Desktop, Tablet)                     |
| `region`               | User's region (North, South, East, West)                  |
| `bounce`               | Whether the user bounced (0 = no, 1 = yes)                |
| `visitor_type`         | Whether the user is New or Returning                      |

---

## 🧹 Data Preprocessing

- **Missing Values**: Filled using median (numeric) or mode (categorical)
- **Outlier Removal**: Applied to selected features using IQR method
- **Feature Engineering**:
  - `avg_time_per_page = time_on_product_page / (page_views + 1)`
- **Standardization**: Applied using `StandardScaler` for numerical columns
- **One-Hot Encoding**: For `device`, `region`, and `visitor_type`

---

## 📊 Visualizations & Insights

A series of visualizations are generated to support storytelling and pattern discovery:

| Chart | Description |
|-------|-------------|
| `correlation_matrix.png` | Correlation heatmap of numerical variables |
| `duration_by_visitor.png` | Boxplot of product page time by visitor type |
| `purchases_by_region.png` | Bar plot of average purchases across regions |
| `duration_vs_purchases.png` | Scatter plot of time vs purchases |
| `bounce_rate_by_visitor.png` | Bar plot showing bounce rate by visitor type |
| `purchases_by_visitor_chart.json` | JSON file for rendering interactive charts |

### ✨ Key Insights
1. Returning visitors spend significantly more time on product pages than new users.
2. Regions show variation in purchase behavior, which may guide targeted marketing.
3. Bounce rates are noticeably higher for new users – indicating onboarding or UX challenges.
4. Moderate positive correlation between product page engagement and purchases.
5. Outliers in engagement metrics were removed to ensure accurate interpretation.

---

## 🧰 Technologies Used

- **Python 3.x**
- **Pandas**, **NumPy** – data manipulation
- **Matplotlib**, **Seaborn** – visualizations
- **Scikit-learn** – preprocessing
- **JSON** – for exportable chart configurations
- Optional: Can be extended into **Streamlit** or **Dash** dashboards

---

## 🗃️ File Outputs

| File                          | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `cleaned_user_behavior.csv`   | Cleaned dataset for modeling or reporting            |
| `eda_insights.txt`            | Summary of analytical insights                       |
| `*.png` charts                | Static plots for reporting or presentation           |
| `purchases_by_visitor_chart.json` | Chart data for integration in dashboards         |

---
**Clone the repository**:
   ```bash
   git clone https://github.com/Arun/User-Detection-Analysis.git
   cd User-Detection-Analysis
```
## 🏁 Next Steps (Optional Enhancements)

- Build a **classification model** to predict purchase likelihood.
- Create a **dashboard** using Streamlit or Dash.
- Integrate real-world behavioral datasets for more accurate modeling.
- Perform **segmentation analysis** (e.g., k-means clustering).
- Deploy in a web app for internal analytics use.

---

## 📜 License

This project is for educational and demonstration purposes. You may use and adapt the code with attribution.

---

## 🙌 Acknowledgements
Special thanks to the Data Analytics faculty and reviewers for their constructive feedback and review-based improvements.


