import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

world_cups_matches = pd.read_csv(r"C:\Users\User\Desktop\WorldCupMatches.csv")
#모든 데이터값이 Nan인 행 삭제
world_cups_matches = world_cups_matches.dropna(how='all')

#preprocessing
world_cups_matches = world_cups_matches.replace('Germany FR', 'Germany')
world_cups_matches = world_cups_matches.replace('Soviet Union', 'Russia')
world_cups_matches = world_cups_matches.replace("C�te d'Ivoire", "Cote d'Ivoire")
world_cups_matches = world_cups_matches.replace('rn">Bosnia and Herzegovina', 'Bosnia and Herzegovina')
world_cups_matches = world_cups_matches.replace('rn">Republic of Ireland', 'Republic of Ireland')
world_cups_matches = world_cups_matches.replace('rn">Serbia and Montenegro', '>Serbia and Montenegro')
world_cups_matches = world_cups_matches.replace('rn">Trinidad and Tobago', '>Trinidad and Tobago')
world_cups_matches = world_cups_matches.replace('rn">United Arab Emirates', '>United Arab Emirates')
#중복 데이터 제거
world_cups_matches = world_cups_matches.drop_duplicates()

#각 팀 별 Home, Away 득점 그룹화
home = world_cups_matches.groupby(['Home Team Name'])['Home Team Goals'].sum()
away = world_cups_matches.groupby(['Away Team Name'])['Away Team Goals'].sum()

#home과 away를 열 방향(axis=1)으로 병합
goal_per_country = pd.concat([home, away], axis = 1).fillna(0)

#각 나라 별 종합 득점 수 계산
goal_per_country['Total Goals'] = goal_per_country['Home Team Goals'] + goal_per_country['Away Team Goals']

pd.set_option('display.max_rows', None)
print(goal_per_country[['Total Goals']])
pd.reset_option('display.max_rows')

#scaling
goal_per_country_scaled = pd.DataFrame(MinMaxScaler().fit_transform(goal_per_country[['Total Goals']]), columns=['Total Goals'])
goal_per_country_scaled['Team'] = goal_per_country.index

#Total Goals 기준으로 내림차순 정렬
goal_per_country_scaled = goal_per_country_scaled.sort_values(by='Total Goals', ascending=False)

plt.bar(goal_per_country_scaled['Team'], goal_per_country_scaled['Total Goals'])
plt.xlabel('Team')
plt.ylabel('Scaled Total Goals')
plt.title('Total Goals Scaled for Each Team')
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.show()