import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

Data = pd.read_csv(r"C:\Users\User\Desktop\Data.csv")

#회귀 학습용 데이터(결측값 제거)
train_df = Data[['Age', 'Salary']].dropna()
X = train_df[['Age']]
y = train_df['Salary']

#회귀 모델 학습 (Salary = a * Age + b)
model = LinearRegression()
model.fit(X, y)
a = model.coef_[0]
b = model.intercept_

#결측값 채우기
for idx, row in Data.iterrows():
    if pd.isna(row['Salary']) and pd.notna(row['Age']):
        Data.at[idx, 'Salary'] = a * row['Age'] + b
    elif pd.isna(row['Age']) and pd.notna(row['Salary']):
        Data.at[idx, 'Age'] = (row['Salary'] - b) / a

#scaling
scaler = MinMaxScaler()
Data[['Age', 'Salary']] = scaler.fit_transform(Data[['Age', 'Salary']])

#regression line 그리기
x_range = np.linspace(Data['Age'].min(), Data['Age'].max(), 100).reshape(-1, 1)
x_range_orig = scaler.inverse_transform(np.hstack([x_range, np.zeros_like(x_range)]))[:, 0].reshape(-1, 1)
y_pred = model.predict(x_range_orig)
y_pred_scaled = scaler.transform(np.hstack([x_range_orig, y_pred.reshape(-1, 1)]))[:, 1]

#purchased 값에 따라 색상 구분
plt.figure(figsize=(10, 6))
plt.scatter(Data['Age'], Data['Salary'], c=(Data['Purchased'] == 'Yes'), label='Data')
plt.plot(x_range, y_pred_scaled, color='black', linewidth=2, label='Regression Line')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Salary')
plt.title('Scaled Age vs Salary with Regression Line')
plt.legend()
plt.show()