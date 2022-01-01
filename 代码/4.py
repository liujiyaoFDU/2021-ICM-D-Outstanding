import numpy as np
import pandas as pd
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 1 + 0.5 * cos
    return sim

def o_sim(x1,x2):
  distance=np.linalg.norm(x1-x2)
  return 1/(1+distance)

def make_global_distance(x1,x2,dis_type):
  if dis_type=='cos':
    return cos_sim(x1,x2)
  elif dis_type=='o':
    return o_sim(x1,x2)
  else:
    return cos_sim(x1,x2)+o_sim(x1,x2)

def count_songs_sim(songs1,songs2):
  all_sim=0
  for i in range(songs2.shape[0]):
    song2=songs2.iloc[i]
    sims=0
    for j in range(songs1.shape[0]):
      song1=songs1.iloc[j]
      sim=make_global_distance(song1,song2,'both')
      sims+=sim
    sims=sims/songs1.shape[0]
    all_sim+=sims
  return all_sim/songs2.shape[0]


genres=['Avant-Garde','Blues',"Children's",'Classical',
    'Comedy/Spoken','Country','Easy Listening','Electronic',
    'Folk','International','Jazz','Latin','New Age','Pop/Rock',
    'R&B;','Reggae','Religious','Stage & Screen','Vocal']
music=pd.read_csv('/content/drive/MyDrive/dataset/2021D/full_music_data.csv')
def make_label(x):
    x=x[1:-1]
    x=x.split(',')
    if len(x)==1:
      return 0
    else:
      return 1

music['label']=music['artists_id'].apply(make_label)
music=music[music['label']==0]

def make_artists(x):
    x=x[1:-1]
    x=x.split(',')
    return x[0]

music['artists_id']=music['artists_id'].apply(make_artists)
max_trend=-3
trend_sum=-3


follower_name=influence['follower_id'].unique().tolist()
names=[]
for name in follower_name:
    info=music[music['artists_id']==str(name)]
    if info.shape[0]>30:
      names.append(name)

for name in names:
    inf_names=influence[influence['follower_id']==name]['influencer_id'].unique().tolist()
    for inf_name in inf_names:
      inf_music=music[music['artists_id']==str(inf_name)]
      follow_music=music[music['artists_id']==str(name)]
      if inf_music.shape[0]<=2 :
        continue
      trend,t_sum=count_trend(follow_music,inf_music)
      if trend>max_trend:
        max_name=name
        max_trend=trend
        max_sum=t_sum
        max_inf_name=inf_name
      if t_sum>trend_sum:
        sum_name=name
        trend_sum=t_sum
        sum_trend=trend
        sum_inf_name=inf_name
effect_df=pd.DataFrame({'influencer':[90124,90124],'follower':[753507,961234],'sum':[0.00033410413828238207,0.0055388490872166265],'max':[0.01251946863297489,0.005492869927425945]})
effect_df['sub']=effect_df['sum']-effect_df['max']
effect_df.to_csv('影响.csv')


def count_trend(follow_music, inf_music):
    window = 4
    stridde = 2
    slidding_sims = []
    feature_columns = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key', 'acousticness',
                       'instrumentalness', 'liveness', 'speechiness', 'duration_ms']

    for i in range(0, follow_music.shape[0] - 4, 2):
        slidding_window = follow_music.iloc[i:i + 4]

        slidding_window = slidding_window
        max_sim = 0
        for j in range(0, inf_music.shape[0] - 2, 1):
            inf_window = inf_music.iloc[j:j + 2]
            sims = count_songs_sim(inf_window[feature_columns], slidding_window[feature_columns])
            if sims > max_sim:
                max_sim = sims
        slidding_sims.append(max_sim)
    slidding_sims_trends = []
    for i in range(len(slidding_sims) - 1):
        trends = (slidding_sims[i + 1] - slidding_sims[i]) / slidding_sims[i]
        slidding_sims_trends.append(trends)
    trend_sum = np.array(slidding_sims_trends).sum()
    max_trend = np.array(slidding_sims_trends).max()
    return max_trend, trend_sum

