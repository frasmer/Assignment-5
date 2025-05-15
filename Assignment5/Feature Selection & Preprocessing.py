import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier

#Part 1: Data Preparation
#Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

#display:
#Number of samples
print("Number of samples:", X.shape[0])
#Feature names
print("\nFeature names:", list(X.columns))
#Target class distribution
print("\nTarget class distribution:")
for idx, count in y.value_counts().items():
    print(f"{data.target_names[idx]} ({idx}): {count} samples")

#Normalize all feature values using MinMaxScaler. 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


#Part 2: Feature Selection Techniques
#A. Chi-Square Test

#Apply SelectKBest with chi2 to score all features.
#모든 특성에 대해 통계적 테스트를 적용해서 점수를 게산하고 상위 k개(이 경우 선택하지 않음)의 특성을 선택
#score_func=chi2 -> 카이제곱 통계량을 기준으로 점수를 계산
#chi2 -> 분류 문제에서 각 feature와 target간의 관계성을 측정
selector = SelectKBest(score_func=chi2, k='all')
#selector에 데이터를 넣어 fitting : chi2 점수 계산
#점수가 높을수록 더 중요한 feature
selector.fit(X_scaled, y)
chi2_scores = selector.scores_

#각 feature의 chi2점수를 df로 생성, Chi2점수를 기준으로 내림차순 정렬
chi2_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Chi2 Score': chi2_scores
}).sort_values(by='Chi2 Score', ascending=False)

#Plot a bar chart of Chi-Square scores.
plt.figure(figsize=(12, 6))
plt.bar(chi2_df['Feature'], chi2_df['Chi2 Score'])
plt.xticks(rotation=90)
plt.title("Chi2 Scores of Features")
plt.ylabel("Chi2 Score")
plt.tight_layout()
plt.show()

#Identify the top 5 features.
print("\nTop 5 Features by Chi-Square Test")
print(chi2_df.head(5))

#B. Lasso Regression
#Use Lasso(alpha=0.01) to fit the scaled data.
#Lasso는 불필요한 feature의 계수를 0으로 만드는 regression 방법
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)
#Lasso모델을 통해 coefficient(중요도와 비슷)를 가져옴
lasso_coefficients = lasso.coef_

#각 feature의 Coefficient를 df로 생성
lasso_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': lasso_coefficients
})
#Coefficient가 0이 아닌(중요도가 0이 아닌) feature들을 따로 저장
non_zero_lasso = lasso_df[lasso_df['Coefficient'] != 0]

#Print out the coefficients.
print("\nLasso Coefficients")
print(lasso_df)
#Identify which features are selected (non-zero).
print("\nLasso Coefficients (non-zero)")
print(non_zero_lasso)

#C. Tree-Based Model
#Train ExtraTreesClassifier on the same data.
#개별 decision tree를 무작위로 만들어 학습
model = ExtraTreesClassifier()
model.fit(X_scaled, y)
#importances는 해당 feature가 예측 정확도 향상에 기여한 정도를 표현
#중요도는 0~1 사이의 값이며, 전체 feature의 중요도 합은 1
importances = model.feature_importances_

#각 feature의 Importance를 df로 생성, Importance를 기준으로 내림차순 정렬
tree_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

#Plot feature importances.
plt.figure(figsize=(12, 6))
plt.bar(tree_df['Feature'], tree_df['Importance'])
plt.xticks(rotation=90)
plt.title("Feature Importances from ExtraTreesClassifier")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

#Identify the top 5 most important features.
print("\ntop 5 most important features")
print(tree_df.head(5))

#Part 3: Comparison & Reflection
#worst concave points가 3개의 방법 모두에서 중요한 feature로, 
#worst radius, worst perimeter, mean concavity, mean concave points가 2개의 방법에서 중요한 feature로 언급되었다다
#->해당 변수들이 모델의 성능과 해석에 있어 핵심적인 feature
#하나의 기법만 사용하는 것이 아닌, 여러 기법을 사용하여 교차 검증하는 것이 중요함함

#Which features were selected consistently across methods?
#3개의 방법에서 모두 언급된 feature : worst concave points
#2개의 방법에서 언급된 feature : worst radius, worst perimeter, mean concavity, mean concave points
#Did any method eliminate features that another considered important?
#Chi-Square나 Tree-based에서 중요하다고 판단된 feature(worst perimeter, mean concavity 등)제거
#Which method do you think is most trustworthy for this task, and why? 
#Tree-based Method :
#비선형 관계를 반영할 수 있음 -> 실제 데이터에 적합
#importance를 점수로 표현하여 중요도를 직관적을 알기 쉬움