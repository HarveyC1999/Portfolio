# Product Conversion Experiment Analysis

## 📌 Overview

This project analyzes the impact of a redesigned checkout flow on user conversion and revenue performance using A/B testing methodology.

The objective is to determine whether the new experience improves business metrics and should be rolled out to all users.

---

## 🧠 Business Problem

An e-commerce platform redesigned its checkout experience to reduce friction and improve purchase conversion.

This analysis evaluates whether the redesigned flow leads to statistically significant improvements in:

- Conversion rate
- Revenue per user
- User engagement

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- SQL

---

## 📂 Project Structure

product-conversion-experiment/

│

├── data/

├── notebooks/

├── sql/

├── dashboard/

├── README.md


---
## 📊 Dataset

The dataset contains simulated experiment data including:

User group (control / variant)
Conversion status
Revenue
Device type
Country
Session duration

Total records: 10,000+

---

##　📈 Analysis Workflow

1. Experiment Metrics

Primary metric:

- Conversion Rate

Secondary metrics:

- Revenue per User
- Session Time

--- 

2. Conversion Analysis

Conversion rate was calculated for both control and variant groups.

Example:

Group	Conversion Rate
Control	12%
Variant	15%

---

3. Statistical Testing

A two-proportion z-test was performed to validate whether the observed difference was statistically significant.

The null hypothesis assumes no difference between groups.

---

4. Segmentation Analysis

Conversion performance was analyzed across device types:

Mobile
Desktop

Desktop users showed the strongest improvement in conversion rate following the checkout redesign.
Results showed significantly stronger uplift among desktop users compared to mobile users.

--- 

## 📊 Key Findings
- Variant group achieved higher conversion rate
- Revenue per user increased in the experiment group
- Statistical testing confirmed significance (p < 0.05)
- Desktop users showed the strongest conversion uplift after the redesign
- Mobile users showed relatively smaller improvement, suggesting the mobile experience may already have been optimized

---

## ✅ Recommendation

- Prioritize rollout for desktop users where the experiment demonstrated the strongest impact
- Continue optimizing the mobile experience through additional UX experiments
- Monitor post-rollout revenue and engagement metrics

--- 
## 🚀 Future Improvements
- Add retention analysis
- Add cohort analysis
- Deploy dashboard for real-time monitoring
- Expand segmentation analysis by region

---

## 👤 Author
Harvey Chang
