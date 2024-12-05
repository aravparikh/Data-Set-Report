# Install Necessary Libraries
!pip install kagglehub pandas scikit-learn xgboost matplotlib seaborn

# Import Required Libraries
import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor, plot_importance

# 1. Download Dataset
path = kagglehub.dataset_download("mexwell/student-scores")
print("Path to dataset files:", path)

csv_file = os.path.join(path, "student-scores.csv")
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print(df.head())  # Display the first few rows
else:
    print("CSV file not found in the dataset path:", path)

# 2. Calculate Average Score
df["average_score"] = df[
    [
        "math_score", "history_score", "physics_score", 
        "chemistry_score", "biology_score", "english_score", 
        "geography_score"
    ]
].mean(axis=1)

# Drop Individual Subject Scores
columns_to_drop = [
    "math_score", "history_score", "physics_score", 
    "chemistry_score", "biology_score", "english_score", 
    "geography_score"
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# 3. One-Hot Encode 'career_aspiration'
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
career_aspiration_encoded = enc.fit_transform(df[['career_aspiration']])
encoded_columns = [f"career_aspiration_{category}" for category in enc.categories_[0]]
encoded_df = pd.DataFrame(career_aspiration_encoded, columns=encoded_columns)
df = pd.concat([df, encoded_df], axis=1)

# 4. Preprocess Features and Target
X = df.drop(['average_score', 'id', 'first_name', 'last_name', 'email', 'gender'], axis=1)
y = df['average_score']
scaler = StandardScaler()
X_numeric = X.select_dtypes(include=['number'])
X_scaled = scaler.fit_transform(X_numeric)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# 6. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
grid_search = GridSearchCV(XGBRegressor(random_state=0), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_scaled, y)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 7. Feature Importance Plot
best_model = grid_search.best_estimator_
plt.figure(figsize=(10, 8))
plot_importance(best_model, importance_type='weight', title="XGBoost Feature Importance")
plt.show()

# 8. Correlation Matrix Heatmap
correlation_matrix = df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix, annot=True, cmap='viridis', fmt=".2f", 
    linewidths=1, cbar_kws={'shrink': 0.8, 'aspect': 20}
)
plt.title("Enhanced Correlation Matrix Heatmap", fontsize=20, fontweight='bold', pad=20)
plt.xticks(rotation=45, fontsize=12, fontweight='bold')
plt.yticks(rotation=0, fontsize=12, fontweight='bold')
plt.xlabel("Subjects", fontsize=14, labelpad=10, fontweight='bold')
plt.ylabel("Subjects", fontsize=14, labelpad=10, fontweight='bold')
plt.tight_layout()
plt.show()

# 9. Bar Plot for Filtered Career Aspirations
df_filtered = df[df.eq(1).any(axis=1)]
df_filtered.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Index')
plt.ylabel('Career Aspirations (1.0 = Chosen)')
plt.title('Career Aspirations (Ones Only)')
plt.tight_layout()
plt.show()
#The headers were made by AI 
