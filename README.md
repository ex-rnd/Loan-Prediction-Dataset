# 🤹 Loan Approval Prediction 

## 📽️ Project Overview
This project builds a full machine‑learning pipeline to predict loan approval outcomes using the Kaggle Loan Prediction Dataset.
The pipeline includes:
-	Data loading & inspection
-	Cleaning (missing values, duplicates, type fixes)
-	Encoding categorical variables
-	Feature scaling (Standardization & Normalization)
-	Exploratory visualizations
-	Train–test splitting
-	Handling class imbalance
-	Training multiple ML models
-	Evaluation using accuracy, classification reports, confusion matrices, and ROC curves

It covers the entire workflow from data loading to model evaluation across SVM, Random Forest, and XGBoost.

### 🧰 1. Setup & Dependencies
#### Install required libraries:
```
pip install imbalanced-learn scikit-learn xgboost pandas numpy matplotlib seaborn
```
Imported libraries include:
- `pandas`,  `numpy`
- `matplotlib`,  `seaborn`
- `sklearn` (preprocessing, model selection, metrics)
- `imbalanced-learn` (SMOTE)
- - `xgboost` (XGBClassifier)

### 📥 2. Data Sourcing
The dataset is loaded from Google Drive:
```
df = pd.read_csv("/content/drive/MyDrive/JengaLabs/loan-prediction-dataset.csv")
```
Initial inspection includes:
- First 5 rows
- Data types
- Shape
- Summary statistics


### 🧹 3. Data Preprocessing
#### 3.1 Missing Values
Missing values were identified and rows with missing entries were dropped:
```
df_clean = df.dropna()
```
Rows with missing values were dropped. Dropping was chosen because only a small number of rows were affected.

#### 3.2 Duplicate Values
Duplicates were detected and removed.
```
duplicate_count = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
```

#### 3.3 Label Encoding
Categorical columns were encoded manually:
```
df_clean.replace({"Gender":{'Male':1,'Female':0}}, inplace=True)
df_clean.replace({"Loan_Status":{'N':0,'Y':1}}, inplace=True)
...
```

Encoded columns include:
- Gender
- Loan_Status
- Self_Employed
- Married
- Property_Area
- Education
- Dependents


#### 3.4 Feature Scaling
##### 3.4.1 Standardization (StandardScaler) — Technique 1
Applied to numerical columns:
```
scaler = StandardScaler()
df_standard_clean[numerical_cols] = scaler.fit_transform(...)
```

##### 3.4.2 Normalization (MinMaxScaler) — Technique 2
Alternative scaling option:
```
scaler = MinMaxScaler()
df_minmax_clean[numerical_cols] = scaler.fit_transform(...)
```


### 📊 4. Exploratory Data Analysis
Visualizations included:
- Education vs Loan Status countplot
- Married vs Loan Status countplot


### 🔀 5. Train–Test Split
Performed using:
```
train_test_split(..., stratify=y)
```
Stratification preserved class balance.


### ⚖️ 6. Handling Class Imbalance
SMOTE was prepared:
```
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```
Random Forest also used `class_weight='balanced'`.


### 🤖 7. Model Training
Three models were trained:
#### 7.1 SVM (Linear Kernel)
```
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_svm_train, y_train)
```

#### 7.2 Random Forest 
```
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
```

#### 7.3 XGBoost
```
xgb_classifier = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='logloss'
)
```


### 📈 8. Model Evaluation
Metrics computed: 
- Accuracy
- Classification report
- Confusion matrix
- ROC curves (binary & multiclass handling)

Example:
```
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```
Random Forest and XGBoost typically give higher accuracy…


### 🧠 9. Interpretation of Results
Key insights:
-	Tree‑based models (RF, XGB) outperform linear SVM on this dataset.
-	Class imbalance affects false positives/negatives.
-	ROC AUC is more informative than accuracy for imbalanced data.


### 📁 11. Project Structure
```
├── data/
│   └── loan-prediction-dataset.csv
├── notebooks/
│   └── Loan_Prediction_Dataset.ipynb
├── README.md
└── LICENSE
```

## 🤝 Contributing
### 🚀 Suggested next steps and improvements
-	Factor notebook logic into testable modules under src/ and add unit tests in tests/.

### 🧭 Style and process
- Tests should import functions from  src/ rather than executing notebook cells..

Thank you for your contributions 🎉

