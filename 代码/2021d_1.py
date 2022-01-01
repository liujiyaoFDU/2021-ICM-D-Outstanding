import pandas as pd
import numpy as np

'''第一问'''
influence=pd.read_csv('/content/drive/MyDrive/dataset/2021D/influence_data.csv')
influence['year']=influence['follower_active_start']-influence['influencer_active_start']
from collections import defaultdict
import math

groups = influence.groupby('influencer_id')
influence_dict = {}
influencer_name=influence['influencer_id'].unique().tolist()
name_dict={}
for id in influencer_name:
  name=influence[influence['influencer_id']==id]['influencer_name'].unique()[0]
  name_dict[id]=name

follower_name=influence['follower_id'].unique().tolist()
for id in follower_name:
  name=influence[influence['follower_id']==id]['follower_name'].unique()[0]
  name_dict[id]=name
all_name=list(set(influencer_name).union(set(follower_name)))
data = defaultdict(list)
for name, group in groups:
    follower_id = []
    follower_scores = []
    main = group['influencer_main_genre'].unique()[0]
    origin_year = group['influencer_active_start'].unique()[0]
    for i in range(group.shape[0]):
        info = group.iloc[i]
        follower_id.append(info['follower_id'])
        if info['follower_main_genre'] != main:
            score = 2
        else:
            score = 1
        if info['year'] < 0:
            year = 1.4 * abs(info['year'])
        else:
            year = info['year']
        score = score * math.exp(year / 80)
        follower_scores.append(score)

        data['fluencer'].append(info['follower_id'])
        data['follower'].append(name)
        data['score'].append(score)
    follower_id = np.array(follower_id)
    follower_scores = np.array(follower_scores)
    influence_dict[name] = (follower_id, follower_scores)
data=pd.DataFrame(data)
data.to_csv('边权重.csv')

influence_score_dict={}
influence_follow_dict={}
n=len(all_name)
for i in range(n):
  name=all_name[i]
  if name in influence_dict.keys():
    follower_id,follower_scores=influence_dict[name]
    influence_follow_dict[name]=follower_id
    influence_score_dict[name]=follower_scores.sum()
  else:
    influence_score_dict[name]=0
    influence_follow_dict[name]=[]

n = len(all_name)
scores = np.zeros(n)
for i in range(n):
    name = all_name[i]

    follows = influence_follow_dict[name]
    score = 0
    if len(follows) == 0:
        continue
    else:
        for follow in follows:
            score += influence_score_dict[follow]
        score += influence_score_dict[name]
        scores[i] = score

n=len(all_name)
all_scores=defaultdict(list)
for i in range(n):
  name=all_name[i]
  name=name_dict[name]
  all_scores['name'].append(name)
  score=scores[i]
  all_scores['score'].append(score)
all_scores=pd.DataFrame(all_scores)
all_scores.to_csv('综合分数.csv')