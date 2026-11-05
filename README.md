# 🌾 Intelligent Crop Recommendation System Using Machine Learning for Precision Agriculture

### Module 6: Deployment, Validation, Monitoring & Maintenance

| Field | Detail |
| :--- | :--- |
| **Course** | FUSE AI-201 — AI Model Development and Implementation |
| **Group Number** | 11 |
| **Module** | 6 — Deploy & Validate |
| **Predecessors** | Module 4 Proposal (approved 26 Sep 2026) · Module 5 README (submitted Oct 2026) |
| **Submission Date** | November 2026 |
| **Primary Model** | Random Forest Classifier — 99.55% test accuracy |
| **Live Application** | 🌐 **[Try the Crop Recommendation App](https://crop-recomm-app.streamlit.app)** |

---

## 🔗 Continuity Statement

**Module 4** defined the problem, target users, dataset, ML task, success criteria, and deployment roadmap. *Note: §5/§8 specified Optuna for hyperparameter tuning; Module 6 implementation reconciles this to GridSearchCV with documented rationale in Residual Items & Scope Reconciliation.*

**Module 5** delivered the trained Random Forest champion, XGBoost challenger, baseline comparison, Optuna tuning of the challenger, SHAP analysis, and a locally-hosted Streamlit prototype.

**Module 6** delivers public deployment, validation against the Module 4 §6 success criteria, and monitoring / maintenance plans for sustained operation.

This document does not repeat Module 5 analysis — it extends and validates it, with scope changes documented in *Residual Items & Scope Reconciliation*.

---

## 🌍 Project Overview

Agriculture contributes approximately 32% of Ethiopia's GDP, employs 64% of the population, and generates 79% of national export earnings. Smallholders cultivate 95% of arable land and produce roughly 90% of agricultural output — yet most lack access to soil testing and rely on blanket national fertilizer recommendations rather than site-specific guidance.

This gap has measurable consequences:
* Ethiopian wheat yields average 2.7 t/ha vs. 5 t/ha achievable at research stations.
* Suboptimal input use costs an estimated 295–946 million Birr annually.
* Precision agriculture technologies deliver an average 22.3% ROI and 18.5% net profit uplift.

This project delivers an end-to-end 22-class crop recommendation system that takes seven soil and climatic inputs (N, P, K, temperature, humidity, pH, rainfall) and returns a recommended crop with a confidence score. Module 6 validates and deploys this system.

> ⚠️ **Pilot status:** The training dataset is India-specific. This system is positioned as a validated pilot and requires Ethiopian local calibration before production use (see *Localization Strategy*).

---

## ❗ Problem Statement

Smallholder farmers cultivate crops not ideally suited to their specific soil chemistry and microclimate, causing:
* Reduced yields and lower productivity
* Inefficient fertilizer application and higher input costs
* Increased economic risk for farming families

**Objective:** Provide an AI-driven, site-specific crop recommendation engine supporting data-driven agricultural planning and precision farming.

---

## 👥 Target Users

| User Group | Role | Value Delivered |
| :--- | :--- | :--- |
| **Farmers (primary)** | Smallholder producers | Localized crop recommendations for seasonal planning |
| **Agricultural advisors** | Extension workers | Evidence-based advisory support tool |
| **Policymakers** | Strategic planners | Data-driven regional agricultural insights |

---

## 📊 Dataset Description

| Attribute | Detail |
| :--- | :--- |
| **Source** | Kaggle — Crop Recommendation Dataset (Aingle, 2020) |
| **Records** | 2,200 observations |
| **Features** | 7 numerical (N, P, K, temperature, humidity, pH, rainfall) |
| **Target** | `label` — 22 crop classes |
| **Class Balance** | Perfectly balanced — 100 samples per class |
| **Geographic origin** | India (general tropical/subtropical) |

### Feature Dictionary

| Feature | Description | Unit |
| :--- | :--- | :--- |
| **N** | Nitrogen content | mg/kg |
| **P** | Phosphorus content | mg/kg |
| **K** | Potassium content | mg/kg |
| **temperature** | Ambient temperature | °C |
| **humidity** | Relative humidity | % |
| **ph** | Soil acidity / alkalinity | 0–14 |
| **rainfall** | Precipitation | mm |

*Quality checks (Module 5): 0 missing values · 0 duplicates · shape (2200, 8) · stratified 80/20 split (1,760 / 440 samples).*

---

## 🌍 Localization Strategy (Ethiopian Context)

**Current gap:** The Kaggle dataset reflects South Asian conditions. Several of its 22 classes (jute, pigeon peas, moth beans) are not staple Ethiopian crops, and its environmental ranges do not map cleanly onto Ethiopia's agro-ecological zones.

### Planned calibration pathway

| Zone | Altitude | Priority crops for calibration |
| :--- | :--- | :--- |
| **Bereha** | < 500 m | Date palm, drought-tolerant sorghum |
| **Kolla** | 500–1,500 m | Sorghum, sesame, cotton, maize |
| **Weina Dega** | 1,500–2,300 m | Maize, teff, wheat, haricot bean |
| **Dega** | 2,300–3,200 m | Barley, wheat, faba bean, potato |
| **Wurch** | > 3,200 m | Limited barley, grazing |

*   **Seasonal alignment:** Predictions to be mapped to Kiremt (Jun–Sep), Belg (Feb–May), and Bega (Oct–Jan) regimes rather than an annual rainfall figure.
*   **Local data sources for retraining:** EthioSIS (soil fertility), NMA (weather), CSA (crop statistics), ATA (advisory datasets).
*   **Interface localization:** Amharic and Afaan Oromoo labels, USSD/SMS fallback, ONNX-based offline packaging.

---

## 🚜 Reaching Smallholders — Pilot & Future Paths

### How the pilot reaches smallholders
The Streamlit app is designed to be used primarily by extension agents and rural development workers, who serve as intermediaries between the AI system and the smallholder. An agent opens the app on a smartphone, enters the farmer's soil and climate values, receives a recommendation with confidence score, and delivers it to the farmer in person, by phone, or by SMS.

This indirect reach path is realistic for Ethiopia's context, where:
* Many smallholders lack smartphone access or reliable data.
* Extension agents are the trusted, existing advisory channel.
* Face-to-face delivery builds trust that a raw app cannot.

### Planned direct-reach channels (post-pilot)
To reach smallholders directly — without an intermediary — the following channels are planned:

| Channel | Description | Timeline |
| :--- | :--- | :--- |
| **SMS gateway** | Farmer texts soil values, receives crop recommendation | Post-pilot |
| **USSD service** | Farmer dials a short code, navigates a menu | Requires Ethio Telecom partnership |
| **Voice IVR** | Farmer calls a hotline, hears recommendation in Amharic/Afaan Oromoo | Requires telecom + TTS/STT |
| **Offline mobile app** | ONNX-packaged model embedded in Android app | Post-pilot |
| **WhatsApp/Telegram bot** | Chat-based recommendation | Post-pilot |

**Why direct-reach channels are not in this iteration:** Direct channels require infrastructure and partnerships beyond the scope of an academic pilot (SMS/USSD requires telecom agreement, IVR requires telephony contracts, etc.). These channels are documented here so the reach strategy is clear to reviewers, and so future work can prioritize them once pilot validation and partnerships are in place.

---

## 🏗 System Architecture

```text
User Input (N, P, K, Temperature, Humidity, pH, Rainfall)
      │
      ▼
Input Validation Layer
      │
      ▼
Preprocessing + Label Encoding
      │
      ▼
Random Forest Champion (.pkl)
      │
      ▼
Crop Prediction + Confidence Score
      │
      ▼
Streamlit User Interface 


---




Model Accuracy Precision Recall F1-Score
Random Forest (Champion) 99.55% 99.57% 99.55% 99.55%
XGBoost (Challenger) 98.64% 98.74% 98.64% 98.62%
SVM (RBF kernel, Module 5 iteration) 98.41% 98.52% 98.41% 98.39%
K-Nearest Neighbors (k=5) 97.95% 98.04% 97.95% 97.93%
Logistic Regression 97.27% 97.40% 97.27% 97.25%
Dummy Classifier (chance floor) 4.55% 0.21% 4.55% 0.40%
