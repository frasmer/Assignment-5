import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import matplotlib.pyplot as plt

np.random.seed(42)
n = 150

# Synthetic features
age = np.random.normal(40, 10, n)
income = np.random.normal(60000, 15000, n)
purchases = np.random.exponential(300, n)
clicks = np.random.poisson(5, n)

# Inject missing values
income[5] = np.nan
purchases[10] = np.nan

# Inject outliers
income[7] = 300000
purchases[3] = 5000

#---------------------------------------------------------

#dataframe으로 변경
df_original=pd.DataFrame({
    'Age': age,
    'Income': income,
    'Purchases': purchases,
    'Clicks': clicks 
})

#비교를 위해 original dataframe 카피
df_filled = df_original.copy()
#과제 조건에 따라 Income은 평균, Purchases는 중앙값으로 Nan 채우기
df_filled['Income'] = df_filled['Income'].fillna(df_filled['Income'].mean())
df_filled['Purchases'] = df_filled['Purchases'].fillna(df_filled['Purchases'].median())

#확인을 위해 original과 filled values을 출력력
print("original data income[5] : ", df_original.loc[5, 'Income'])
print("filled mean income[5] : ", df_filled.loc[5, 'Income'])
print("original data purchases[10] : ", df_original.loc[10, 'Purchases'])
print("filled median purchases[10] : ", df_filled.loc[10, 'Purchases'])

#Nan 존재 확인
print("Rows with NaN:\n", df_filled.isna().sum())

#IQR정의
def get_outlier_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

#IQR을 이용해서 outlier 감지
income_lower, income_upper = get_outlier_bounds(df_filled['Income'])
purchases_lower, purchases_upper = get_outlier_bounds(df_filled['Purchases'])

# outlier masks
income_outlier_mask = (df_filled['Income'] < income_lower) | (df_filled['Income'] > income_upper)
purchases_outlier_mask = (df_filled['Purchases'] < purchases_lower) | (df_filled['Purchases'] > purchases_upper)

# Outlier 값과 해당 인덱스 출력
print("Income outliers:")
print(df_filled[income_outlier_mask]['Income'])
print("\nPurchases outliers:")
print(df_filled[purchases_outlier_mask]['Purchases'])

# #outlier를 ffill(바로 앞 값으로 채우기)로 대체하기
# df_filled['Income'] = df_filled['Income'].mask(income_outlier_mask).ffill()
# df_filled['Purchases'] = df_filled['Purchases'].mask(purchases_outlier_mask).ffill()
#outlier clip(임계값을 초과하거나 미달하는 값을 임계값으로 자르는 방식)하기기
df_filled['Income'] = df_filled['Income'].clip(lower=income_lower, upper=income_upper)
df_filled['Purchases'] = df_filled['Purchases'].clip(lower=purchases_lower, upper=purchases_upper)

#a. Min-Max Scaling for age
#데이터 값 중 가장 작은 값을 0, 가장 큰 값을 1로 기준을 정하여 scaling 하는 방법
#나이는 데이터 분포가 균일하고, 범위가 일정한 연속형 변수이기에 Min-Max Scaling 적합
df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_filled[['Age']]), columns=['Age_minmax'])

#b. Z-score Standardization for income
#데이터의 평균을 0, 표준편차를 1로 변환하는 방법
#데이터가 정규분포를 따른다고 가정할 때 적합하며, 평균과 표준편차를 기준으로 스케일링하므로 outlier에 민감
# plt.hist(df_filled['Income'])
# plt.show()
#해당 데이터는 정규분포에 가깝기 때문에 적합한 방법
df_scaled['Income_ZScore'] = StandardScaler().fit_transform(df_filled[['Income']])

#c. Log Transformation for purchases
#X' = log(X + 1)으로 변환_log 변환 시 0이 될 경우를 대비하여 +1
#비대칭적인 분포(positively skewed)를 정규 분포로 변환하기에 구매와 같이 왜곡이 있는 데이터에 적합
# plt.hist(df_filled['Purchases'])
# plt.show()
#해당 데이터 분포에 적합한 방법
df_scaled['Purchases_Log'] = np.log(df_filled[['Purchases']] + 1)

#d. Robust Scaling for income
#median과 IQR을 이용하여 데이터를 변환하는 방법
#outlier에 강한 내성을 갖기에 소득(타당한 outlier-고소득자)과 같은 데이터에 적합
df_scaled['Income_Robust'] = RobustScaler().fit_transform(df_filled[['Income']])

#e. Vector Normalization for [age, income, clicks]
#각 행(row), 즉 각 데이터 포인트 벡터 전체의 크기를 1로 만드는 방법
#여러 feature들(다양한 단위)를 한 벡터로 보고 유사도 계산 등에 활용할 때 적합
df_scaled[['Age_vector', 'Income_vector', 'Clicks_vector']] = Normalizer().fit_transform(df_filled[['Age', 'Income', 'Clicks']])

#Print the transformed values (first 5 entries)
print(df_scaled.head())

#Histogram of purchases before and after log transform
fig, axes = plt.subplots(1, 2)
axes[0].set_title("purchases before log transform")
axes[0].hist(df_filled['Purchases'], bins=20)

axes[1].set_title("purchases after log transform")
axes[1].hist(df_scaled['Purchases_Log'], bins=20)

plt.tight_layout()

#Box plot of income before and after robust scaling
fig, axes = plt.subplots(1, 2)
axes[0].set_title("income before robust scaling")
axes[0].boxplot(df_filled['Income'])

axes[1].set_title("income after robust scaling")
axes[1].boxplot(df_scaled['Income_Robust'])

plt.tight_layout()

plt.show()