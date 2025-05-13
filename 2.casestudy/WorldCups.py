import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

world_cups = pd.read_csv(r"C:\Users\User\Desktop\WorldCups.csv")

#Attendance 단위 구분자 제거
world_cups['Attendance'] = world_cups['Attendance'].str.replace('.', '', regex=False).astype(int)

scaler = StandardScaler()

#역대 월드컵 관중 scaling
scaled_attendance = scaler.fit_transform(world_cups[['Attendance']]).reshape(-1,)

plt.title("All-time World Cup Attendance")
plt.plot(world_cups['Year'], scaled_attendance, marker = "o")

#plt.show()

#역대 월드컵 경기당 득점
#GoalsPerMatch 추가
world_cups['GoalsPerMatch'] = world_cups['GoalsScored']/world_cups['MatchesPlayed']

#scaling
scaled_goalsScored = scaler.fit_transform(world_cups[['GoalsScored']]).reshape(-1,)
scaled_goalsPerMatch = scaler.fit_transform(world_cups[['GoalsPerMatch']]).reshape(-1,)
scaled_matchesPlayed = scaler.fit_transform(world_cups[['MatchesPlayed']]).reshape(-1,)

fig, axes = plt.subplots(1, 2)
axes[0].set_title("World Cup points per game")
axes[0].bar(x=world_cups['Year'], height=scaled_goalsScored)
axes[0].plot(world_cups['Year'], scaled_matchesPlayed, marker="o")

axes[1].set_title("goals scored per game")
axes[1].plot(world_cups['Year'], scaled_goalsPerMatch, marker="o")
 
#plt.show()

#성적 집계
winner = world_cups['Winner']
runners_up = world_cups['Runners-Up']
third = world_cups['Third']
fourth = world_cups['Fourth']

winner_count = pd.Series(winner.value_counts())
runners_up_count = pd.Series(runners_up.value_counts())
third_count = pd.Series(third.value_counts())
fourth_count = pd.Series(fourth.value_counts())

ranks = pd.DataFrame({
  "Winner" : winner_count,
  "Runners_Up" : runners_up_count,
  "Third" : third_count,
  "Fourth" : fourth_count
})
#4강 이내 순위 기록이 없는 나라들은 0으로 처리
ranks = ranks.fillna(0).astype('int64')

#scaling
ranks_minmax_scaled = pd.DataFrame(MinMaxScaler().fit_transform(ranks), columns=ranks.columns)

plt.title("National grade tally")
ranks_minmax_scaled.plot.bar()

plt.show()