# Employee Attrition Prediction and Analysis (EAPA)

![Project Banner](https://via.placeholder.com/800x200.png?text=Employee+Attrition+Prediction+Analysis)  
*Predict employee attrition risks and empower HR with actionable insights.*

---

## 👥 Team Members
- **Ahmed Saeed**
- **Yusuf Amr**
- **Ahmed Ashraf**
- **Khaled Sayed**

---

## 📌 Project Overview
**Objective**: Build a machine learning model to predict employee attrition and enable proactive retention strategies.  
**Impact**:  
- Reduce turnover costs (recruitment, training, lost productivity).  
- Identify high-risk employees for targeted interventions.  

**Project Lifecycle**:  
`Data Collection → EDA → Feature Engineering → Model Development → API Deployment → Monitoring`

---

## 📊 Dataset
- **Source**: [IBM HR Analytics Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
- **Features**: Age, Department, Salary, Job Satisfaction, Tenure, Attrition (Target: `Yes/No`).  
- **Class Imbalance**:  
  - **237 Employees** left ("Yes").  
  - **1233 Employees** stayed ("No").  

---

## 🛠️ Methodology

### 1. Data Preprocessing & EDA
- **Key Steps**:  
  - Analyzed department distribution:  
    ```python
    data['Department'].value_counts()
    ```
  - Checked attrition balance:  
    ```python
    data['Attrition'].value_counts()
    ```
- **Libraries**: `pandas`, `matplotlib`, `seaborn`.

### 2. Feature Engineering
- **New Features**:  
  - `Salary/Tenure Ratio`: Salary divided by years of tenure.  
  - `Overtime Impact`: Derived from business travel frequency.  
- **Statistical Tests**:  
  - T-test confirmed **significant salary differences** between employees who left vs. stayed.

### 3. Model Development
- **Algorithms**:  
  | Model               | Recall | AUC-ROC |  
  |---------------------|--------|---------|  
  | Logistic Regression | 78%    | 0.84    |  
  | **Random Forest**   | 85%    | 0.92    |  
  | XGBoost             | 83%    | 0.89    |  

- **Handling Imbalance**:  
  - Applied **SMOTE** to oversample the minority class:  
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    ```

### 4. Deployment & MLOps
- **Flask API**:  
  ```python
  @app.route('/predict', methods=['POST'])
  def predict():
      employee_data = request.get_json()
      prediction = model.predict(employee_data)
      return jsonify({"attrition_risk": prediction.tolist()})
