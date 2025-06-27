# E‑Commerce Recommendation Engine

A scalable recommendation engine built with PySpark and MLFLow on Databricks, designed for a fictional fashion & beauty brand. It uses collaborative filtering and product embeddings to deliver personalized product suggestions and improve customer engagement.

## 🎯 Problem Statement

1. **Information overload**
   Customers struggle to find relevant items among thousands of SKUs.
2. **Low conversion rates**
   Generic browsing experiences fail to resonate, hurting sales.
3. **Poor user retention**
   Lack of personalization fails to incentivize repeat visits.
4. **High return rates**
   Irrelevant recommendations lead to dissatisfaction and returns.

## 💡 Business Objectives

* **Boost Average Order Value (AOV):** Suggest complementary or upsell products.
* **Increase customer satisfaction:** Provide relevant, engaging recommendations.
* **Reduce churn:** Personalized experiences encourage repeat shopping.
* **Improve inventory utilization:** Promote less popular items to balance demand.

## 🛠 Key Features & Architecture

1. **Data Ingestion & ETL**

   * Raw purchase logs (user‑item‑rating) ingested via Spark pipelines.
   * Transformed and cached for model training.

2. **Collaborative Filtering (ALS)**

   * Spark ML’s Alternating Least Squares for scalable recommendations.
   * Handles sparse interactions typical of retail datasets.

3. **Content-based / Hybrid Extensions**

   * Incorporates product metadata (descriptions, categories) using embeddings.
   * Enables recommendations for new (cold-start) items.

4. **Model Tracking with MLflow**

   * Stores training artifacts, performance metrics, and parameters.
   * Facilitates A/B testing and model governance.

5. **Deployment on Databricks**

   * Production-ready jobs schedule pipelines and inference endpoints.
   * Supports near-real-time feature generation and scoring.

## 📈 Business Benefits

| Goal                      | Outcome                                      |
| ------------------------- | -------------------------------------------- |
| Personalized discovery    | Customers receive relevant item suggestions  |
| Higher engagement         | Increased click-through and conversion rates |
| Enhanced product exposure | Broader catalog visibility and lower returns |
| Scalable infrastructure   | Engine scales with data, reducing latency    |

## ⚙️ Getting Started

1. **Clone repo**

   ```bash
   git clone https://github.com/Lwhieldon/E-CommerceRecommendationEngine.git
   cd E-CommerceRecommendationEngine
   ```

2. **Setup Databricks & MLflow**

   * Launch workspace, configure cluster, attach repo.
   * Install Python dependencies via `requirements.txt`.

3. **Run pipeline notebooks**

   * `01_data_prep.ipynb`: ETL and data preprocessing
   * `02_model_train.ipynb`: ALS training, hyperparameter tuning, MLflow logging
   * `03_inference_service.ipynb`: Score and generate recommendations

4. **Deploy to production**

   * Convert notebooks to jobs on Databricks, schedule via MLflow/Jobs UI.
   * Expose RESTful API for real-time recommendations.

## ✅ How to Evaluate

* **Offline metrics:** RMSE, Precision\@K, Recall\@K.
* **A/B tests:** Compare CTR and AOV with control.
* **Business KPIs:** Measure reclamation rate, repeat purchase frequency.

## 🚀 Next Steps

* Integrate transformer-based text embeddings to enhance product content understanding ([github.com][1], [linkedin.com][2], [medium.com][3], [blog.adnansiddiqi.me][4]).
* Add session-based modeling (e.g. via graph or embedding techniques) similar to recent Amazon co-purchase systems .
* Deploy front-end widget to surface real-time recommendations on site.
* Implement feedback loop to refine models based on live interactions.

---

### 📞 Contact

Lee Whieldon – Data Strategist & Engineer
📍 Baltimore, MD · github.com/Lwhieldon

---

### References

1: https://github.com/oaslananka/E-Commerce-Recommendation-System?utm_source=chatgpt.com "oaslananka/E-Commerce-Recommendation-System - GitHub"

2: https://www.linkedin.com/pulse/predicting-your-needs-recommendation-systems-e-commerce-qeccf?utm_source=chatgpt.com "Predicting Your Needs: Recommendation Systems in E-commerce"

3: https://medium.com/the-ai-guide/e-commerce-recommendation-engine-with-collaborative-filtering-cb19cd542c18?utm_source=chatgpt.com "E-Commerce Recommendation Engine with Collaborative Filtering"

4: https://blog.adnansiddiqi.me/building-an-e-commerce-product-recommendation-system-with-openai-embeddings-in-python/?utm_source=chatgpt.com "Building an E-commerce Product Recommendation System with ..."
