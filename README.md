Here is the complete README.md content, ready to copy directly into a file named README.md:

```markdown
# Intelligent Crop Recommendation System Using Machine Learning for Precision Agriculture

**FUSE AI-201: AI Model Development and Implementation — Final Project**

**Group Number:** 11

**Notebook:** [Google Colab Implementation](https://colab.research.google.com/drive/19ywbAHZtikl0vflVMKyUf8KuK3a2LI2F)

**Deployment:** Streamlit Web Application

**Primary Model:** Random Forest Classifier (99.55% Test Accuracy)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Target Users](#target-users)
4. [Dataset Description](#dataset-description)
5. [System Architecture](#system-architecture)
6. [Data Preparation & Quality Assessment](#data-preparation--quality-assessment)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Model Development & Comparison](#model-development--comparison)
9. [Optimization & Interpretability](#optimization--interpretability)
10. [System Testing & Validation](#system-testing--validation)
11. [Streamlit Application (Prototype)](#streamlit-application-prototype)
12. [Project Directory Structure](#project-directory-structure)
13. [Installation & Setup](#installation--setup)
14. [Limitations & Future Improvements](#limitations--future-improvements)
15. [Team Contributions](#team-contributions)
16. [Group Reflection](#group-reflection)
17. [Submission Checklist](#submission-checklist)

---

## Project Overview

Agriculture serves as the foundation of Ethiopia's economy, supporting a large portion of the population and driving national growth. However, smallholder farmers often rely on traditional intuition and generalized farming guidelines due to limited access to affordable soil testing and localized advisory services.

This project delivers an end-to-end **Intelligent Crop Recommendation System** using Machine Learning. Formulated as a **22-class multi-class classification problem**, the model takes measured soil nutrients (Nitrogen, Phosphorus, Potassium) and environmental indicators (Temperature, Humidity, Soil pH, Rainfall) to accurately recommend the most suitable crop along with an inference confidence score.

---

## Problem Statement

Smallholder farmers frequently cultivate crops that are not ideally suited to their specific soil chemistry and microclimate. This challenge causes:

- Reduced crop yields and lower agricultural productivity
- Inefficient fertilizer application and higher operational costs
- Increased economic risk and lower profit margins for farming families

**Objective:** Develop an AI-driven, site-specific crop recommendation engine that supports data-driven agricultural planning and precision farming practices.

---

## Target Users

- **Primary Users:**
  - Smallholder farmers planning seasonal sowing activities.
  - Agricultural policymakers requiring data-driven insights rather than purely political assumptions.
- **Secondary Users:**
  - Agricultural extension workers and rural development agents.
  - Agronomists, agricultural consultants, and development organizations.

---

## Dataset Description

- **Source:** Kaggle Crop Recommendation Dataset
- **Records:** 2,200 observations
- **Features:** 7 continuous numerical features
- **Target:** `label` (Crop Category)
- **Target Classes (22):** Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee.
- **Class Balance:** Perfectly balanced (100 samples per crop class).

### Feature Dictionary

| Feature Name | Description | Unit |
| :--- | :--- | :--- |
| `N` | Nitrogen content ratio in soil | mg/kg |
| `P` | Phosphorus content ratio in soil | mg/kg |
| `K` | Potassium content ratio in soil | mg/kg |
| `temperature` | Ambient air temperature | °C |
| `humidity` | Relative atmospheric humidity | % |
| `ph` | Soil acidity / alkalinity | pH scale (0–14) |
| `rainfall` | Precipitation depth | mm |

---

## System Architecture

```text
User / Field Measurement Inputs
(N, P, K, Temperature, Humidity, pH, Rainfall)
│
▼
Input Validation Layer
│
▼
Data Preprocessing & Scaling
│
▼
Trained Random Forest Classifier Engine
│
▼
Crop Classification & Confidence Score
│
▼
Streamlit User Interface Output
```

---

Data Preparation & Quality Assessment

1. Quality Check:
   · Missing Values: df.isnull().sum() confirmed 0 missing entries.
   · Duplicate Records: df.duplicated().sum() confirmed 0 duplicates.
   · Dataset Dimensions: (2200, 8).
2. Preprocessing:
   · Label Encoding: Multi-class string labels converted into numerical categories.
   · Feature Scaling: StandardScaler applied where appropriate for distance-based baselines.
   · Data Partitioning: Stratified 80/20 train-test split (1,760 training samples, 440 testing samples) ensuring equal representation across all 22 classes.

---

Exploratory Data Analysis (EDA)

· Nutrient Variance: Nitrogen, Phosphorus, and Potassium show wide distributions across crop categories.
· Environmental Boundaries: Rainfall values span arid (~20 mm) to high-precipitation environments (~300 mm).
· Soil pH: Exhibits a normal distribution centered at pH 6.47, consistent with arable agricultural soils.
· Correlation: Low overall multicollinearity among most variables, with a moderate positive correlation observed between Phosphorus (P) and Potassium (K).
· Crop Threshold Insights:
  · Rice requires high rainfall thresholds (> 180 mm).
  · Grapes and Apples exhibit distinct dependencies on high Potassium (K) levels.
  · Tree-based algorithms are optimal for capturing these hard environmental thresholds.

---

Model Development & Comparison

Multiple models were trained, evaluated, and compared on the unseen 20% test partition:

Model Category Accuracy Precision Recall F1-Score
Random Forest Classifier Advanced (Ensemble) 99.55% 99.57% 99.55% 99.55%
XGBoost Classifier Advanced (Boosting) 99.09% 99.15% 99.09% 99.08%
K-Nearest Neighbors (KNN) Baseline 97.95% --- --- ---
Logistic Regression Baseline 97.27% --- --- ---
Dummy Classifier Control Baseline 5.23% --- --- ---

Model Selection: The Random Forest Classifier was selected as the final production model due to its superior overall accuracy (99.55%), robustness against overfitting, and high F1-score across all 22 classes.

---

Optimization & Interpretability

Hyperparameter Tuning (Optuna)

Optimization was conducted on gradient boosting models with the following top-performing parameters:

· max_depth: 8
· n_estimators: 83
· learning_rate: 0.194

Model Interpretability (SHAP Analysis)

Feature importance calculated via Shapley Additive Explanations (SHAP) ranked input significance:

1. Humidity
2. Rainfall
3. Potassium (K)
4. Nitrogen (N)
5. Phosphorus (P)

Climatic variables (humidity and rainfall) play the most critical role in separating broad crop families, while soil nutrients resolve specific crop types within similar climates.

---

System Testing & Validation

Functional test cases verified the end-to-end inference engine:

Test Case Inputs Expected / Predicted Crop Confidence Result
Test Case 1: Normal Conditions N=90, P=42, K=43, Temp=20.8°C, Hum=82.1%, pH=6.5, Rain=202.9mm Maize 98.0% Passed ✅
Test Case 2: Extreme Drought Temp=48.0°C, Rain=5.0mm, N=20, P=40, K=20, Hum=45%, pH=7.0 Moth Beans 41.5% Passed ✅

---

Streamlit Application (Prototype)

A responsive web application (app.py) provides an interactive interface for extension agents and farmers:

· Real-time recommendation delivery within milliseconds.
· Input validation rules with bounded sliders and numerical entry fields.
· Prediction confidence visualization.
· Responsible AI Advisory Notice: Explicitly reminds operators that output serves as advisory decision-support rather than a replacement for local field testing.

---

Project Directory Structure

```text
├── app.py                          # Streamlit web application
├── Crop_recommendation.xlsx        # Dataset (2,200 records)
├── crop_recommendation_model.pkl   # Serialized model and encoders
├── requirements.txt                # Project package dependencies
├── IMPLEMENTATION_ASSIGNMENT_PAPER.docx  # Final implementation report
├── crop_recommendation.ipynb       # Implementation notebook
└── README.md                       # Project documentation
```

---

Installation & Setup

1. Prerequisites

· Python 3.9, 3.10, or 3.11 installed.
· git version control installed.

2. Clone the Repository

```bash
git clone https://github.com/<your-username>/crop-recommendation-system.git
cd crop-recommendation-system
```

3. Create and Activate a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the Application

```bash
streamlit run app.py
```

---

Limitations & Future Improvements

Limitations

1. Soil physical properties (texture, sand/silt/clay ratio, bulk density) are not yet integrated.
2. Topographic variables (altitude, slope, aspect) are omitted.
3. The dataset reflects general tropical/subtropical conditions; local Ethiopian agro-ecological zones (Dega, Weina Dega, Kolla) need direct calibration.
4. Recommendations provide technical suitability rather than real-time economic market viability.

Future Roadmap

· Localized Datasets: Train on regional soil survey data from the Ethiopian Ministry of Agriculture.
· Automated Weather API: Integrate live GPS-based weather data retrieval (Temperature, Rainfall, Humidity).
· Offline Mobile Version: Package as a lightweight Android / USSD application for rural field workers.
· Economic Forecasting: Combine agronomic suitability with seasonal market crop price projections.

---

Team Contributions

Team Member Core Contributions
Mihret Endeshaw Problem identification, literature review, technical reporting
Biniyam Meldeye Problem identification, background study, documentation
Eyuel Getachew Problem formulation, baseline model evaluation, report drafting
Samson Ambissa Dataset acquisition, data cleaning, exploratory data analysis
Yonas Debole Preprocessing pipeline, feature scaling, label encoding validation
Fikerachen Sisay Advanced model training (Random Forest, XGBoost), SHAP analysis
Sisay Yeshak Streamlit application development, UI layout design
Milkoma Ayele Web prototype architecture, input validation layer
Temesgen Teshome Streamlit app integration, test case execution, verification
Boniyas Wagari Quality assurance, boundary testing, drought scenario validation

---

Group Reflection

· Key Learning: Learned how to execute the entire AI development lifecycle—from problem scoping and data hygiene to model interpretability (SHAP) and production deployment.
· Biggest Challenge: Managing data consistency and label encoding parity between model training, evaluation scripts, and the real-time inference web app.
· Resolution: Implemented an integrated preprocessing and label-encoder serialization pipeline to ensure zero mismatch during inference.
· Future Enhancement: Given more time, our priority would be ingesting localized soil test samples from Ethiopian farming regions and embedding real-time weather forecasting APIs.
· Teamwork Reflection: Structured role allocation across data engineering, model development, deployment, and testing allowed rapid parallel execution.

---

Submission Checklist

☑ Implementation Notebook (.ipynb / Colab Link)
☑ Project Report Document (IMPLEMENTATION ASSIGNMENT PAPER.docx)
☑ Comprehensive README.md
☑ Dataset File (Crop_recommendation.xlsx)
☑ Dependency Manifest (requirements.txt)
☑ Serialized Model Artifact (crop_recommendation_model.pkl)
☑ Working Web Application (app.py)
☑ Full AI Lifecycle Demonstrated (Problem → Data → Analysis → Model → Evaluation → Testing → Working Prototype)

```

**To use this:**
1. Copy everything inside the code block above.
2. Paste it into a new file named `README.md` (plain text, not `.docx`).
3. Save and upload to your project folder/repository.

> **Note:** The entire content is wrapped in a single outer code fence so you can copy it as one block. Be careful to **not** copy the outermost triple backticks — only the content inside. The inner code fences (for `text` and `bash` blocks) should remain.# README.md

Intelligent Crop Recommendation System Using Machine Learning for Precision Agriculture

**FUSE AI-201: AI Model Development and Implementation — Final Project**  
**Group Number:** 11  
**Notebook:** [Google Colab Implementation](https://colab.research.google.com/drive/19ywbAHZtikl0vflVMKyUf8KuK3a2LI2F)  
**Deployment:** Streamlit Web Application  
**Primary Model:** Random Forest Classifier (99.55% Test Accuracy)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Target Users](#target-users)
4. [Dataset Description](#dataset-description)
5. [System Architecture](#system-architecture)
6. [Data Preparation & Quality Assessment](#data-preparation--quality-assessment)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Model Development & Comparison](#model-development--comparison)
9. [Optimization & Interpretability](#optimization--interpretability)
10. [System Testing & Validation](#system-testing--validation)
11. [Streamlit Application (Prototype)](#streamlit-application-prototype)
12. [Project Directory Structure](#project-directory-structure)
13. [Installation & Setup](#installation--setup)
14. [Limitations & Future Improvements](#limitations--future-improvements)
15. [Team Contributions](#team-contributions)
16. [Group Reflection](#group-reflection)
17. [Submission Checklist](#submission-checklist)

---

## Project Overview

Agriculture serves as the foundation of Ethiopia's economy, supporting a large portion of the population and driving national growth. However, smallholder farmers often rely on traditional intuition and generalized farming guidelines due to limited access to affordable soil testing and localized advisory services.

This project delivers an end-to-end **Intelligent Crop Recommendation System** using Machine Learning. Formulated as a **22-class multi-class classification problem**, the model takes measured soil nutrients (Nitrogen, Phosphorus, Potassium) and environmental indicators (Temperature, Humidity, Soil pH, Rainfall) to accurately recommend the most suitable crop along with an inference confidence score.

---

## Problem Statement

Smallholder farmers frequently cultivate crops that are not ideally suited to their specific soil chemistry and microclimate. This challenge causes:
* Reduced crop yields and lower agricultural productivity
* Inefficient fertilizer application and higher operational costs
* Increased economic risk and lower profit margins for farming families

**Objective:** Develop an AI-driven, site-specific crop recommendation engine that supports data-driven agricultural planning and precision farming practices.

---

## Target Users

* **Primary Users:**
  * Smallholder farmers planning seasonal sowing activities.
  * Agricultural policymakers requiring data-driven insights rather than purely political assumptions.
* **Secondary Users:**
  * Agricultural extension workers and rural development agents.
  * Agronomists, agricultural consultants, and development organizations.

---

## Dataset Description

* **Source:** Kaggle Crop Recommendation Dataset
* **Records:** 2,200 observations
* **Features:** 7 continuous numerical features
* **Target:** `label` (Crop Category)
* **Target Classes (22):** Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee.
* **Class Balance:** Perfectly balanced (100 samples per crop class).

### Feature Dictionary

| Feature Name | Description | Unit |
| :--- | :--- | :--- |
| `N` | Nitrogen content ratio in soil | mg/kg |
| `P` | Phosphorus content ratio in soil | mg/kg |
| `K` | Potassium content ratio in soil | mg/kg |
| `temperature` | Ambient air temperature | °C |
| `humidity` | Relative atmospheric humidity | % |
| `ph` | Soil acidity / alkalinity | pH scale (0–14) |
| `rainfall` | Precipitation depth | mm |

---

## System Architecture

```text
               User / Field Measurement Inputs
         (N, P, K, Temperature, Humidity, pH, Rainfall)
                             │
                             ▼
                   Input Validation Layer
                             │
                             ▼
                 Data Preprocessing & Scaling
                             │
                             ▼
             Trained Random Forest Classifier Engine
                             │
                             ▼
            Crop Classification & Confidence Score
                             │
                             ▼
                Streamlit User Interface Output
```

---

## Data Preparation & Quality Assessment

1. **Quality Check:**
   * Missing Values: `df.isnull().sum()` confirmed **0** missing entries.
   * Duplicate Records: `df.duplicated().sum()` confirmed **0** duplicates.
   * Dataset Dimensions: `(2200, 8)`.
2. **Preprocessing:**
   * **Label Encoding:** Multi-class string labels converted into numerical categories.
   * **Feature Scaling:** `StandardScaler` applied where appropriate for distance-based baselines.
   * **Data Partitioning:** Stratified 80/20 train-test split (1,760 training samples, 440 testing samples) ensuring equal representation across all 22 classes.

---

## Exploratory Data Analysis (EDA)

* **Nutrient Variance:** Nitrogen, Phosphorus, and Potassium show wide distributions across crop categories.
* **Environmental Boundaries:** Rainfall values span arid (~20 mm) to high-precipitation environments (~300 mm).
* **Soil pH:** Exhibits a normal distribution centered at `pH 6.47`, consistent with arable agricultural soils.
* **Correlation:** Low overall multicollinearity among most variables, with a moderate positive correlation observed between Phosphorus (P) and Potassium (K).
* **Crop Threshold Insights:**
  * Rice requires high rainfall thresholds (> 180 mm).
  * Grapes and Apples exhibit distinct dependencies on high Potassium (K) levels.
  * Tree-based algorithms are optimal for capturing these hard environmental thresholds.

---

## Model Development & Comparison

Multiple models were trained, evaluated, and compared on the unseen 20% test partition:

| Model | Category | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **Advanced (Ensemble)** | **99.55%** | **99.57%** | **99.55%** | **99.55%** |
| XGBoost Classifier | Advanced (Boosting) | 99.09% | 99.15% | 99.09% | 99.08% |
| K-Nearest Neighbors (KNN) | Baseline | 97.95% | — | — | — |
| Logistic Regression | Baseline | 97.27% | — | — | — |
| Dummy Classifier | Control Baseline | 5.23% | — | — | — |

**Model Selection:** The **Random Forest Classifier** was selected as the final production model due to its superior overall accuracy (99.55%), robustness against overfitting, and high F1-score across all 22 classes.

---

## Optimization & Interpretability

### Hyperparameter Tuning (Optuna)
Optimization was conducted on gradient boosting models with the following top-performing parameters:
* `max_depth`: 8
* `n_estimators`: 83
* `learning_rate`: 0.194

### Model Interpretability (SHAP Analysis)
Feature importance calculated via Shapley Additive Explanations (SHAP) ranked input significance:
1. **Humidity**
2. **Rainfall**
3. **Potassium (K)**
4. **Nitrogen (N)**
5. **Phosphorus (P)**

Climatic variables (humidity and rainfall) play the most critical role in separating broad crop families, while soil nutrients resolve specific crop types within similar climates.

---

## System Testing & Validation

Functional test cases verified the end-to-end inference engine:

| Test Case | Inputs | Expected / Predicted Crop | Confidence | Result |
| :--- | :--- | :---: | :---: | :---: |
| **Test Case 1: Normal Conditions** | `N=90, P=42, K=43, Temp=20.8°C, Hum=82.1%, pH=6.5, Rain=202.9mm` | **Maize** | 98.0% | **Passed ✅** |
| **Test Case 2: Extreme Drought** | `Temp=48.0°C, Rain=5.0mm, N=20, P=40, K=20, Hum=45%, pH=7.0` | **Moth Beans** | 41.5% | **Passed ✅** |

---

## Streamlit Application (Prototype)

A responsive web application (`app.py`) provides an interactive interface for extension agents and farmers:
* Real-time recommendation delivery within milliseconds.
* Input validation rules with bounded sliders and numerical entry fields.
* Prediction confidence visualization.
* **Responsible AI Advisory Notice:** Explicitly reminds operators that output serves as advisory decision-support rather than a replacement for local field testing.

---

## Project Directory Structure

```text
├── app.py                             # Streamlit web application
├── Crop_recommendation.xlsx           # Dataset (2,200 records)
├── crop_recommendation_model.pkl      # Serialized model and encoders
├── requirements.txt                   # Project package dependencies
├── IMPLEMENTATION_ASSIGNMENT_PAPER.docx # Final implementation report
├── crop_recommendation.ipynb          # Implementation notebook
└── README.md                          # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.9, 3.10, or 3.11 installed.
* `git` version control installed.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/crop-recommendation-system.git
cd crop-recommendation-system
```

### 3. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## Limitations & Future Improvements

### Limitations
1. Soil physical properties (texture, sand/silt/clay ratio, bulk density) are not yet integrated.
2. Topographic variables (altitude, slope, aspect) are omitted.
3. The dataset reflects general tropical/subtropical conditions; local Ethiopian agro-ecological zones (Dega, Weina Dega, Kolla) need direct calibration.
4. Recommendations provide technical suitability rather than real-time economic market viability.

### Future Roadmap
* **Localized Datasets:** Train on regional soil survey data from the Ethiopian Ministry of Agriculture.
* **Automated Weather API:** Integrate live GPS-based weather data retrieval (Temperature, Rainfall, Humidity).
* **Offline Mobile Version:** Package as a lightweight Android / USSD application for rural field workers.
* **Economic Forecasting:** Combine agronomic suitability with seasonal market crop price projections.

---

## Team Contributions

| Team Member | Core Contributions |
| :--- | :--- |
| **Mihret Endeshaw** | Problem identification, literature review, technical reporting |
| **Biniyam Meldeye** | Problem identification, background study, documentation |
| **Eyuel Getachew** | Problem formulation, baseline model evaluation, report drafting |
| **Samson Ambissa** | Dataset acquisition, data cleaning, exploratory data analysis |
| **Yonas Debole** | Preprocessing pipeline, feature scaling, label encoding validation |
| **Fikerachen Sisay** | Advanced model training (Random Forest, XGBoost), SHAP analysis |
| **Sisay Yeshak** | Streamlit application development, UI layout design |
| **Milkoma Ayele** | Web prototype architecture, input validation layer |
| **Temesgen Teshome** | Streamlit app integration, test case execution, verification |
| **Boniyas Wagari** | Quality assurance, boundary testing, drought scenario validation |

---

## Group Reflection

* **Key Learning:** Learned how to execute the entire AI development lifecycle—from problem scoping and data hygiene to model interpretability (SHAP) and production deployment.
* **Biggest Challenge:** Managing data consistency and label encoding parity between model training, evaluation scripts, and the real-time inference web app.
* **Resolution:** Implemented an integrated preprocessing and label-encoder serialization pipeline to ensure zero mismatch during inference.
* **Future Enhancement:** Given more time, our priority would be ingesting localized soil test samples from Ethiopian farming regions and embedding real-time weather forecasting APIs.
* **Teamwork Reflection:** Structured role allocation across data engineering, model development, deployment, and testing allowed rapid parallel execution.

---

## Submission Checklist

- [x] Implementation Notebook (`.ipynb` / [Colab Link](https://colab.research.google.com/drive/19ywbAHZtikl0vflVMKyUf8KuK3a2LI2F))
- [x] Project Report Document (`IMPLEMENTATION ASSIGNMENT PAPER.docx`)
- [x] Comprehensive `README.md`
- [x] Dataset File (`Crop_recommendation.xlsx`)
- [x] Dependency Manifest (`requirements.txt`)
- [x] Serialized Model Artifact (`crop_recommendation_model.pkl`)
- [x] Working Web Application (`app.py`)
- [x] Full AI Lifecycle Demonstrated (Problem → Data → Analysis → Model → Evaluation → Testing → Working Prototype)