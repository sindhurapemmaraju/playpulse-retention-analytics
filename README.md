# 🎮 PlayPulse — Google Play Store Retention Intelligence Pipeline

> **Live Dashboard:** [playpulse.streamlit.app](https://picturebycurls-28lcilj5u7pfj23pwkv9xq.streamlit.app)

An end-to-end data science and analytics pipeline analyzing **9,658 Google Play Store apps** to answer one core business question:

> *"What makes a mobile app retain users — and can we predict it?"*

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack & Why](#tech-stack--why)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Phase 1 — ETL Pipeline](#phase-1--etl-pipeline)
6. [Phase 2 — EDA](#phase-2--exploratory-data-analysis)
7. [Phase 3 — Hypothesis Testing](#phase-3--hypothesis-testing)
8. [Phase 4 — Regression](#phase-4--regression)
9. [Phase 5 — ML Classification](#phase-5--ml-classification)
10. [Phase 6 — K-Means Clustering](#phase-6--k-means-clustering)
11. [Phase 7 — Gemini AI Insight Layer](#phase-7--gemini-ai-insight-layer)
12. [Key Findings](#key-findings)
13. [Interview Q&A](#interview-qa)
14. [How to Run](#how-to-run)

---

## Project Overview

PlayPulse is a **production-style analytics pipeline** built to demonstrate end-to-end data science skills — from raw data ingestion to AI-powered insights. It covers every stage a data scientist at Google, Amazon, or Microsoft would be expected to execute:

| Stage | What it covers |
|-------|---------------|
| Data Engineering | ETL pipeline, BigQuery warehouse, cloud storage |
| Exploratory Analysis | Statistical summaries, Plotly visualizations |
| Statistical Testing | Hypothesis tests, effect sizes, p-values |
| Regression | Linear Regression vs Random Forest Regressor |
| Classification | Logistic Regression vs Random Forest Classifier |
| Clustering | K-Means segmentation with Elbow + Silhouette method |
| AI Layer | Gemini 2.5 Flash API for automated cluster narratives |
| Deployment | Streamlit app deployed on Streamlit Cloud |

---

## Tech Stack & Why

| Tool | Why we used it |
|------|---------------|
| **Python** | Industry standard for data science |
| **Google Colab** | Cloud notebook — no local setup, shareable |
| **BigQuery** | Industry-standard data warehouse; SQL at scale — mirrors real DS workflows at Google/Amazon |
| **Pandas / NumPy** | Data manipulation and numerical computation |
| **Scikit-Learn** | ML models — Random Forest, Logistic Regression, KMeans, StandardScaler |
| **Plotly** | Interactive visualizations — better than Matplotlib for dashboards |
| **SciPy** | Statistical hypothesis testing — t-tests, Mann-Whitney U, Pearson correlation |
| **Gemini 2.5 Flash API** | AI-powered cluster narrative generation via prompt engineering |
| **Streamlit** | Rapid dashboard deployment — used by data teams at Airbnb, Uber, Google |
| **GitHub** | Version control and portfolio presentation |

---

## Architecture

```
Raw CSV (Kaggle)
      ↓
Python Cleaning (Pandas)
      ↓
BigQuery Warehouse (Google Cloud)
      ↓
EDA → Hypothesis Testing → Regression → Classification → Clustering
      ↓
Gemini AI Insight Layer
      ↓
Streamlit Dashboard (Live)
```

**Why this architecture?**
This mirrors a real production data pipeline. Raw data is never worked on directly — it flows through a cleaning stage, gets stored in a warehouse, and is queried for analysis. This separation of concerns is standard practice at any data-driven company.

---

## Dataset

**Source:** Google Play Store Apps dataset (Kaggle — Lavanya)
**Size:** 10,841 raw records → 9,658 after cleaning
**Features:** App name, Category, Rating, Reviews, Size, Installs, Type (Free/Paid), Price, Content Rating, Last Updated

**Why this dataset?**
- Large enough to be statistically credible
- Messy enough to demonstrate real cleaning skills
- Contains Indian-relevant categories (Finance, Food, Shopping, Social, Communication) enabling a culturally-grounded hypothesis
- Reflects a real business problem — app retention is a core metric at every mobile product company

---

## Phase 1 — ETL Pipeline

**What we did:**
1. Loaded raw CSV and identified data quality issues
2. Dropped duplicates (10,841 → 9,660 unique apps)
3. Cleaned messy columns:
   - `Installs`: removed "+" and "," → converted to integer (`10,000+` → `10000`)
   - `Price`: removed "$" → converted to float
   - `Size`: converted "19M" → `19.0` MB, "Varies with device" → NaN
   - `Last Updated`: parsed to datetime → engineered `Days_Since_Update`
4. Filled Rating nulls with **category median** (not overall median — smarter because Finance app ratings ≠ Game ratings)
5. Flagged Indian-relevant categories: FINANCE, FOOD_AND_DRINK, SHOPPING, COMMUNICATION, SOCIAL
6. Loaded clean data into **BigQuery** via `load_table_from_dataframe`

**Why category median for null filling?**
If you fill Rating nulls with the overall median (4.3), you're assuming a Medical app with no rating is as good as the average Game. Category median respects domain context — a Medical app's missing rating is better estimated from other Medical apps.

**Result:** 9,658 clean rows, 0 nulls, all correct dtypes, stored in BigQuery

---

## Phase 2 — Exploratory Data Analysis

**5 key visualizations built with Plotly:**

### Finding 1 — More apps ≠ more installs
FAMILY has the most apps (1,831) but only 2.4M avg installs. GAME has fewer apps (959) but 14.4M avg installs. Category crowding doesn't drive success.

### Finding 2 — Rating distribution is left-skewed (survivorship bias)
Almost no apps below 3.0. Massive spike between 4.0–4.5. Mean = 4.19, Median = 4.30. **Why?** Bad apps get uninstalled and eventually delisted — what remains in the store is artificially high-rated. This is survivorship bias in action.

### Finding 3 — Free vs Paid overlap heavily
Paid apps (mean 4.27) slightly outrate free (mean 4.18) but boxes overlap almost completely. Needed formal testing.

### Finding 4 — Indian vs Global categories look similar
Indian-relevant category mean: 4.18. Global mean: 4.19. Visually indistinguishable. Needed formal testing.

### Finding 5 — Installs and Rating are independent
Trendline on Rating vs Log(Installs) scatter is essentially flat for both Free and Paid apps. Popularity does not equal quality on the Play Store.

---

## Phase 3 — Hypothesis Testing

### Why hypothesis testing?
EDA shows you what things *look* like. Hypothesis testing tells you whether what you see is *real* or just random noise. This is the difference between an analyst and a data scientist.

### Test 1 — Free vs Paid Ratings
- **Test used:** Mann-Whitney U (non-parametric — ratings aren't normally distributed)
- **Result:** p < 0.0001 ✅ Statistically significant
- **But:** Difference is only 0.09, identical medians (4.30)
- **Conclusion:** Statistically significant but **practically negligible**. Pricing model alone doesn't drive user satisfaction.
- **Why this matters:** This is the statistical significance vs practical significance distinction — a critical concept interviewers test at MAANG companies.

### Test 2 — Indian-Relevant vs Global Category Ratings
- **Test used:** Mann-Whitney U + Cohen's d effect size
- **Result:** p = 0.3027 ❌ Not significant. Cohen's d = -0.016 (negligible)
- **Conclusion:** Indian-category apps rate identically to global apps. Despite behavioral differences in how Indian users interact with technology, Play Store satisfaction signals are universal.
- **Why this matters:** We hypothesized a difference based on prior knowledge of Indian consumer behavior — the data said no. Honest null results are real data science.

### Test 3 — Update Recency vs Rating
- **Test used:** Pearson correlation
- **Result:** r = -0.116, p < 0.0001 ✅ Significant negative correlation
- **Pattern:** Clean monotonic decline across all 5 recency bins:
  - Updated < 1 month: 4.248
  - Updated 1+ year: 4.095
- **Conclusion:** Apps updated more recently consistently rate higher. Active maintenance signals quality to users.

---

## Phase 4 — Regression

**Goal:** Predict exact app rating from structural features

**Features used:** Log(Installs), Log(Reviews), Size_MB, Price, Days_Since_Update, Is_Indian_Category, Category_Encoded, Content_Rating_Encoded, Type_Encoded

**Why log-transform Installs and Reviews?**
Both are heavily right-skewed (a few apps have billions of installs, most have thousands). Log transformation brings the distribution closer to normal and prevents large values from dominating the model.

### Results

| Model | R² | RMSE |
|-------|-----|------|
| Linear Regression | 0.1130 | 0.4718 |
| Random Forest Regressor | 0.1220 | 0.4693 |

### Why is R² only 0.12?
**This is the most important finding of the regression phase.** Structural app features only explain 12% of rating variance. The other 88% is driven by factors we can't measure in this dataset — actual app quality, UX, performance, customer support, word of mouth. This means **Play Store ratings are driven by intangible quality, not measurable metadata.** That's a genuine product insight.

### Feature Importance (Random Forest)
1. Log_Reviews — 0.267 (most important)
2. Days_Since_Update — 0.241
3. Size_MB — 0.207
4. Category_Encoded — 0.124
5. Is_Indian_Category — 0.014 (near zero — confirms Test 2)

---

## Phase 5 — ML Classification

**Goal:** Predict whether an app is "high retention" or not (binary classification)

**Why classification after regression?**
Predicting an exact rating (regression) is hard — R² = 0.12. But predicting *whether* an app will be high or low retention is a more tractable business question. Binary classification gave us much stronger results.

**Target variable engineering:**
```python
High_Retention = (Rating >= 4.0) AND (Installs >= 10,000)
```
Result: 53% high retention (5,117), 47% low retention (4,541) — near-perfect balance, no need for SMOTE or class weighting.

### Results

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 0.8385 | 0.8618 |
| Precision | 0.8456 | 0.8199 |
| Recall | 0.8506 | 0.9473 |
| F1 Score | 0.8481 | 0.8790 |
| ROC-AUC | 0.9172 | 0.9308 |

**Why Random Forest wins:**
Higher ROC-AUC (0.93 vs 0.92) and dramatically higher Recall (0.947 vs 0.851). For a retention prediction problem, Recall matters more than Precision — you'd rather flag a potentially churning app for intervention (false positive) than miss a genuinely churning app (false negative).

**Why ROC-AUC = 0.93 is strong:**
AUC of 0.5 = random guessing. AUC of 1.0 = perfect. 0.93 means our model correctly ranks a high-retention app above a low-retention app 93% of the time.

---

## Phase 6 — K-Means Clustering

**Goal:** Segment apps into behavioral groups without labels

**Why K-Means?**
We don't have predefined app quality categories. K-Means lets the data reveal its own natural groupings.

**Why StandardScaler before K-Means?**
K-Means uses Euclidean distance. Without scaling, Installs (range: 0–1B) would completely dominate over Rating (range: 1–5) just because of numeric magnitude. StandardScaler puts all features on the same scale.

**Choosing K:**
Used two methods:
- **Elbow Method:** Inertia curve flattens around K=4–6
- **Silhouette Score:** Peaks at K=6 (score = 0.345)
- **Decision:** K=6

### The 6 App Segments

| Segment | Count | Avg Rating | Avg Installs | Key Insight |
|---------|-------|------------|--------------|-------------|
| **Mainstream Champions** | 3,606 | 4.30 | 16.7M | Sweet spot — free, lightweight, actively maintained |
| **Hidden Gems** | 2,998 | 4.38 | 4.4K | Best ratings, near-zero reach — discovery gap |
| **Viral Giants** | 1,114 | 4.27 | 12.2M | Network-effect driven, large apps (65MB avg) |
| **Struggling Survivors** | 910 | 3.08 | 304K | Moderate installs, low satisfaction — UX debt |
| **Abandoned but Alive** | 1,012 | 4.13 | 739K | Not updated in 3+ years, still getting installs |
| **Premium Outliers** | 18 | 3.93 | 12K | $381 avg price, 39% Indian category — B2B tools |

---

## Phase 7 — Gemini AI Insight Layer

**What it does:**
For each of the 6 cluster profiles, we passed structured data to the **Gemini 2.5 Flash API** with a prompt asking for a 3-sentence product analyst insight covering:
1. What type of apps are in this segment?
2. What drives their performance?
3. What should product teams do?

**Why this matters:**
This demonstrates AI integration as a *productivity layer* in analytics — not just ML modeling. Automating narrative generation from cluster data is a real workflow at product analytics teams.

**Prompt engineering approach:**
Each prompt included quantitative cluster statistics as structured context, a role specification ("You are a senior product analyst at Google Play Store"), and explicit output format constraints ("exactly 3 sentences").

---

## Key Findings

1. **Ratings are driven by intangible quality, not metadata** — R² = 0.12 means 88% of rating variance is unexplained by structural features
2. **Active maintenance is the strongest controllable signal** — update recency monotonically correlates with rating (r = -0.116, p < 0.0001)
3. **Indian-category apps rate identically to global apps** — no statistically significant difference (p = 0.30, d = -0.016)
4. **Pricing model doesn't drive satisfaction** — Free vs Paid difference is statistically real but practically negligible (0.09 points)
5. **High vs low retention is very predictable** — ROC-AUC = 0.93 on binary classification
6. **Install volume and review count dominate retention signals** — top 2 features in both regression and classification
7. **Hidden Gems are the Play Store's biggest opportunity** — highest avg rating (4.38) but virtually no installs (4.4K avg)
8. **Abandoned apps are a competitive moat** — 1,012 apps untouched for 3+ years still accumulating 739K avg installs

---

## Interview Q&A

*This section covers questions a recruiter or interviewer might ask about this project.*

**Q: Why did you choose this dataset?**
A: It's large enough to be statistically credible, messy enough to demonstrate real cleaning skills, and the Indian-category angle let me apply domain knowledge about consumer behavior — relevant to my UX research background at Trash to Treasure.

**Q: Why did you use Mann-Whitney U instead of a t-test?**
A: Ratings aren't normally distributed — the histogram shows a left-skewed distribution with a hard ceiling at 5.0. Mann-Whitney U is a non-parametric test that doesn't assume normality, making it more appropriate than a t-test here.

**Q: Your R² is only 0.12 — isn't that a bad model?**
A: Not in this context. A low R² is itself the finding — it tells us that structural app metadata explains very little of rating variance. The real drivers (app quality, UX, performance) aren't in this dataset. Reporting a low R² honestly and interpreting it correctly is better data science than overfitting to get a high number.

**Q: Why Random Forest over Logistic Regression?**
A: Two reasons. First, Random Forest achieved higher ROC-AUC (0.93 vs 0.92) and significantly higher Recall (0.947 vs 0.851). Second, for a retention problem, Recall is the priority metric — missing a churning app is more costly than a false alarm. Random Forest's non-linear decision boundaries also better capture the interaction effects between install count and review volume.

**Q: How did you choose K=6 for clustering?**
A: I used two methods: the Elbow Method (where inertia stops dropping sharply) and Silhouette Score (which measures cluster separation). Both pointed to K=6 — the Silhouette Score peaked at 0.345 for K=6 before dropping significantly at K=7.

**Q: What would you do differently with more time?**
A: Three things. First, add time-series analysis — the dataset is from 2018 and tracking rating changes over update cycles would strengthen the recency finding. Second, add NLP analysis on user reviews to understand *why* apps get low ratings, not just that they do. Third, deploy the ML model as a REST API so app developers could input their app's features and get a retention probability score.

**Q: What does the Indian-category finding mean for product strategy?**
A: It means localization strategy shouldn't assume Indian users rate apps differently — they apply the same satisfaction standards as global users. The differentiation opportunity is in discovery and distribution (getting Indian-relevant apps found), not in quality expectations.

**Q: Why BigQuery instead of just using pandas locally?**
A: Three reasons. First, it mirrors real DS workflows — production data lives in warehouses, not local CSV files. Second, BigQuery's SQL interface is how analytics teams at Google, Amazon, and Microsoft actually query data. Third, it demonstrates cloud fluency, which is a hard skill gap for most student projects.

---

## How to Run

**Option 1 — View the live dashboard:**
[https://picturebycurls-28lcilj5u7pfj23pwkv9xq.streamlit.app](https://picturebycurls-28lcilj5u7pfj23pwkv9xq.streamlit.app)

**Option 2 — Run locally:**
```bash
git clone https://github.com/sindhurapemmaraju/playpulse-retention-analytics
cd playpulse-retention-analytics
pip install -r requirements.txt
streamlit run playpulse_app.py
```

**Option 3 — Explore the notebook:**
Open the Colab notebook to see the full pipeline with outputs.

---

## About

Built by **Sindhura Pemmaraju** — B.Tech Data Science student at Malla Reddy University, Hyderabad (GPA 9.08) | UI/UX Designer & Product Researcher at Trash to Treasure GreenTech Startup

[LinkedIn](https://linkedin.com/in/sindhura-pemmaraju) · [GitHub](https://github.com/sindhurapemmaraju) · [Google Skills Profile](https://skillshop.credential.net/profile/sindhura)
