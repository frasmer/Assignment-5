import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

world_cups_players = pd.read_csv(r"C:\Users\User\Desktop\WorldCupPlayers.csv")

world_cups_players = world_cups_players.replace('M�LLER', 'Müller')
world_cups_players = world_cups_players.replace('PEL� (Edson Arantes do Nascimento)', 'PELÉ (Edson Arantes do Nascimento)')

#Event 결측값 제거 및 list 분리
world_cups_players = world_cups_players.dropna(subset=['Event'])
world_cups_players['Event List'] = world_cups_players['Event'].str.split()
world_cups_players_exploded = world_cups_players.explode('Event List')

#골 이벤트 필터링
goals = world_cups_players_exploded[world_cups_players_exploded['Event List'].str.startswith('G')]

# 선수별 골 횟수 집계
player_goals = goals['Player Name'].value_counts()

#scaling
scaler = MinMaxScaler()
scaled_goals = scaler.fit_transform(player_goals.values.reshape(-1, 1))

#DataFrame 생성 및 득점 기준 정렬
scaled_goals_world_cups_players = pd.DataFrame(scaled_goals, columns=['Scaled Goals'], index=player_goals.index)
scaled_goals_world_cups_players = scaled_goals_world_cups_players.sort_values(by='Scaled Goals', ascending=False)

#일부 선수만 결과 출력(상위 10명)
top_scorers = scaled_goals_world_cups_players.head(10)

plt.bar(top_scorers.index, top_scorers['Scaled Goals'])
plt.title('Top 10 Goal Scorers')
plt.xlabel('Player Name')
plt.ylabel('Scaled Goals')
plt.xticks(rotation=-45)
plt.tight_layout()
plt.show()