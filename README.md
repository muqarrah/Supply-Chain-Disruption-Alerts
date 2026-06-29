# SupplyGuard — Supply Chain Disruption Alert System

SupplyGuard is an end-to-end predictive risk-monitoring platform designed to shift procurement workflows from reactive damage control to proactive asset management. By evaluating real-time operational data and categorical environmental hazard signals, the system outputs a continuous risk score from 0 to 100, allowing supply chain managers to identify upcoming supplier failures before production lines stall.

## 🚀 Core Objective
Traditional supply chains suffer from a "reactive lag"—disruptions are only noticed after a shipment is delayed or fails. SupplyGuard fixes this bottleneck by providing an automated, predictive warning engine that calculates precise risk thresholds, giving companies a strategic lead-time window to coordinate alternate logistics routing.

## 🧠 Machine Learning & Architecture
The predictive core is powered by a **Logistic Regression** classifier optimized to compute disruption probabilities via an S-shaped Sigmoid curve. 

### Feature Space Evaluation
* **Continuous Operational Metrics:** Tracks internal health parameters like the **KG Score** (manufacturing precision tolerances and quality defects) and the **Emergency Score** (facility backlogs or sudden on-site resource scarcities).
* **Categorical Environmental Flags:** Binary indicators (0 or 1) tracking real-time regional or digital anomalies: Twitter Alerts, Weather Risks, Economic Risks, Geo-Political Risks, and Technical Infrastructure Failures.

---

## 📊 Model Performance Summary
The predictive engine was validated against an offline dataset split consisting of 480 training samples and 120 testing samples, achieving exceptional statistical boundaries:

* **Overall Accuracy:** 95.00%
* **Precision (Positive Predictive Value):** 97.30%
* **Recall (Sensitivity / Detection Rate):** 94.74%
* **F1-Score:** 0.9600
* **ROC-AUC Engine:** 0.9913

### Test Set Confusion Matrix
* **True Positives (TP):** 72 (Correctly flagged risky suppliers)
* **True Negatives (TN):** 42 (Correctly identified stable assets)
* **False Positives (FP):** 2 (Safe suppliers flagged as high-risk)
* **False Negatives (FN):** 4 (Risky suppliers missed by the model)

---

## ⚙️ Operational Risk Thresholds
To turn mathematical probability curves into intuitive management decisions, the application splits live risk scores into three operational tiers:

* 🟢 **Low Risk (0.0 to 44.9):** Supplier is completely healthy and stable. Continue standard routine monitoring.
* 🟡 **Medium Risk (45.0 to 69.9):** Emerging operational strain or environmental threats. High priority watch status; begin verifying alternative backup inventories.
* 🔴 **High Risk (70.0 to 100.0):** Critical danger zone. Immediate deployment of contingency sourcing and fallback shipping routes to protect the manufacturing line.

---

## 🛠️ Project Stack & Installation

### Tech Stack
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla ES6)
* **Backend:** Python, Flask API
* **Data Science Suite:** Scikit-Learn, Pandas, NumPy


