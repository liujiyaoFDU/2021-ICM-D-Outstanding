import pandas as pd
import numpy as np
music=pd.read_csv('/content/drive/MyDrive/dataset/2021D/full_music_data.csv')
influence=pd.read_csv('/content/drive/MyDrive/dataset/2021D/influence_data.csv')
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
music.head(5)
artist=pd.read_csv('/content/drive/MyDrive/dataset/2021D/data_by_artist.csv')
influencer_name = influence['influencer_id'].unique().tolist()
genre_dict = {}
for id in influencer_name:
    genres = influence[influence['influencer_id'] == id]['influencer_main_genre'].unique().tolist()
    if len(genres) != 1:
        print('False')
    genre_dict[id] = genres[0]

follower_name = influence['follower_id'].unique().tolist()
for id in follower_name:
    genres = influence[influence['follower_id'] == id]['follower_main_genre'].unique()
    if len(genres) != 1:
        print('False')
    genre_dict[id] = genres[0]

genre_members=defaultdict(list)
genres=['Avant-Garde','Blues',"Children's",'Classical',
    'Comedy/Spoken','Country','Easy Listening','Electronic',
    'Folk','International','Jazz','Latin','New Age','Pop/Rock',
    'R&B;','Reggae','Religious','Stage & Screen','Vocal']

for key,genre in genre_dict.items():
  if genre=='Unknown':
    continue
  genre_members[genre].append(key)

name_list1 = np.random.choice(genre_members['Pop/Rock'], 100, replace=False)
name_list2 = np.random.choice(genre_members['R&B;'], 100, replace=False)
name_list3 = np.random.choice(genre_members['Jazz'], 100, replace=False)
label_dict = {}
for name in name_list1:
    label_dict[name] = 'Pop/Rock'
for name in name_list2:
    label_dict[name] = 'R&B;'

for name in name_list3:
    label_dict[name] = 'Jazz'
feature_columns=['artists_id','danceability','energy','valence','tempo','loudness','mode','key','acousticness','instrumentalness','liveness','speechiness','duration_ms','song_title (censored)']
music_feature=music[feature_columns]

nor_columns=['tempo','loudness','key','duration_ms']
for column in nor_columns:
  music_feature[column]=music_feature[column]/music_feature[column].max()


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
    return cos_sim(x1,x2)/1.5+o_sim(x1,x2)


feature_columns=['artist_id','danceability','energy','valence','tempo','loudness','mode','key','acousticness','instrumentalness','liveness','speechiness','duration_ms']
artist_feature=artist[feature_columns]
nor_columns=['tempo','key','duration_ms']
for column in nor_columns:
  artist_feature.loc[:,column]=artist_feature[column]/artist_feature[column].max()
artist_feature.loc[:,'loudness']=artist_feature['loudness']/25

train_feature=artist_feature.loc[artist_feature['artist_id'].isin(name_list1)]

from tqdm import tqdm

dis_type = 'o'
feature_columns = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness', 'duration_ms']
music_global_sim = defaultdict(list)
music_all_sim = defaultdict(list)
with tqdm(total=train_feature.shape[0] - 1) as pbar:
    for i in range(train_feature.shape[0] - 1):
        pbar.update(1)
        for j in range(i + 1, train_feature.shape[0]):
            info1 = train_feature.iloc[[i, j], :]

            index = info1.index.tolist()
            index1, index2 = index

            x1 = info1.iloc[0]
            id1 = int(x1['artist_id'])
            x1 = x1[feature_columns]
            x2 = info1.iloc[1]
            id2 = int(x2['artist_id'])
            x2 = x2[feature_columns]
            global_distance = make_global_distance(x1, x2, dis_type)
            music_global_sim['id0'].append(id1)
            music_global_sim['genre0'].append(label_dict[id1])
            music_global_sim['id1'].append(id2)
            music_global_sim['genre1'].append(label_dict[id2])
            music_global_sim['sim'].append(global_distance)
            local_distance = make_local_distance(x1, x2, feature_sorts, dis_type)
            distance = (global_distance + local_distance) / 2
            music_all_sim['id0'].append(id1)
            music_all_sim['id1'].append(id2)
            music_all_sim['genre0'].append(label_dict[id1])
            music_all_sim['genre1'].append(label_dict[id2])
            music_all_sim['sim'].append(distance)

music_global_sim = pd.DataFrame(music_global_sim)
music_global_sim.to_csv('全局相似度.csv')
music_all_sim = pd.DataFrame(music_all_sim)
music_all_sim.to_csv('全局+局部相似度.csv')

def make_label(x):
  if int(x) in label_dict.keys():
    return label_dict[int(x)]
  else:
    return 4

music_feature['label']=music_feature['artists_id'].apply(make_label)
artist=pd.read_csv('/content/drive/MyDrive/dataset/2021D/data_by_artist.csv')
artist_feature1=music_feature[music_feature['label']=='Pop/Rock']
print(artist_feature1.shape)
artist_feature2=music_feature[music_feature['label']=='R&B;']
artist_feature3=music_feature[music_feature['label']=='Jazz']
def make_label1(x):
  if len(x)<=5:
    return 0
  else:
    return 1
music_feature['label1']=music_feature['song_title (censored)']
music_feature1=music_feature[music_feature['label1']==0]


artist_feature1=music_feature1[music_feature1['label']=='Pop/Rock']
artist_feature2=music_feature1[music_feature1['label']=='R&B;']
artist_feature3=music_feature1[music_feature1['label']=='Jazz']

index1=artist_feature1.index.tolist()
index1=list(np.random.choice(index1,4,replace=False))

index2=artist_feature2.index.tolist()
index2=list(np.random.choice(index2,4,replace=False))

index3=artist_feature3.index.tolist()
index3=list(np.random.choice(index3,4,replace=False))

index1.extend(index2)
index1.extend(index3)

music_artist=music_feature.loc[index1]

dis_type = 'o'
feature_columns = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness', 'duration_ms']
music_global_sim1 = defaultdict(list)
music_all_sim1 = defaultdict(list)
with tqdm(total=music_artist.shape[0] - 1) as pbar:
    for i in range(music_artist.shape[0] - 1):
        pbar.update(1)
        for j in range(i + 1, music_artist.shape[0]):
            info1 = music_artist.iloc[[i, j], :]

            # index=info1.index.tolist()
            # index1,index2=index

            x1 = info1.iloc[0]
            id1 = x1['song_title (censored)']
            x1 = x1[feature_columns]
            x2 = info1.iloc[1]
            id2 = x2['song_title (censored)']
            x2 = x2[feature_columns]
            global_distance = make_global_distance(x1, x2, dis_type)
            music_global_sim1['id0'].append(id1)
            # music_global_sim1['genre0'].append(id1)
            music_global_sim1['id1'].append(id2)
            music_global_sim1['sim'].append(global_distance)
            music_global_sim1['id0'].append(id2)
            # music_global_sim1['genre0'].append(id1)
            music_global_sim1['id1'].append(id1)
            music_global_sim1['sim'].append(global_distance)

            # music_global_sim1['genre1'].append(label_dict[id2])

            local_distance = make_local_distance(x1, x2, feature_sorts, dis_type)
            distance = (global_distance + local_distance) / 2
            music_all_sim1['id0'].append(id1)
            music_all_sim1['id1'].append(id2)
            # music_all_sim1['genre0'].append(label_dict[id1])
            # music_all_sim1['genre1'].append(label_dict[id2])
            music_all_sim1['sim'].append(distance)
            music_all_sim1['id0'].append(id2)
            music_all_sim1['id1'].append(id1)
            # music_all_sim1['genre0'].append(label_dict[id1])
            # music_all_sim1['genre1'].append(label_dict[id2])
            music_all_sim1['sim'].append(distance)

A = np.ones((12, 12))
music_global_sim1 = pd.DataFrame(music_global_sim1)
for i in range(music_artist.shape[0]):
    for j in range(music_artist.shape[0]):
        if i == j:
            continue
        else:
            info1 = music_artist.iloc[i]
            info2 = music_artist.iloc[j]

            id1 = info1['song_title (censored)']
            id2 = info2['song_title (censored)']

            score = music_global_sim1.loc[(music_global_sim1['id0'] == id1) & (music_global_sim1['id1'] == id2)]['sim']
            A[i, j] = score
columns = music_artist['song_title (censored)'].to_numpy()
A = pd.DataFrame(A)
A.columns = columns
A.index = columns
A.to_csv('歌曲.csv')


def make_label(x):
  if x['genre0']==x['genre1']:
    return 0
  else:
    return 1


id2=music_all_sim['id1'].to_numpy()
id1=music_all_sim['id0'].to_numpy()
sim=music_all_sim['sim'].to_numpy()
gener1=music_all_sim['genre0'].to_numpy()
gener2=music_all_sim['genre1'].to_numpy()

extra={}
extra['id0']=id2
extra['id1']=id1
extra['genre0']=gener2
extra['genre1']=gener1
extra['sim']=sim

extra=pd.DataFrame(extra)
music_all_sim=pd.concat([music_all_sim,extra])
music_all_sim['label']=music_all_sim.apply(make_label,axis=1)


same=music_all_sim[music_all_sim['label']==0]
diff=music_all_sim[music_all_sim['label']==1]
print(same.shape,diff.shape)

pop_same_mean=same[same['genre0']=='Pop/Rock']['sim'].mean()
rb_same_mean=same[same['genre1']=='R&B;']['sim'].mean()
jz_same_mean=same[same['genre1']=='Jazz']['sim'].mean()

pop_rb_mean=diff.loc[(diff['genre0']=='Pop/Rock')&(diff['genre1']=='R&B;')]['sim'].mean()
pop_jz_mean=diff.loc[(diff['genre0']=='Pop/Rock')&(diff['genre1']=='Jazz')]['sim'].mean()
rb_jz_mean=diff.loc[(diff['genre0']=='R&B;')&(diff['genre1']=='Jazz')]['sim'].mean()

pop_same_min=same[same['genre0']=='Pop/Rock']['sim'].min()
rb_same_min=same[same['genre1']=='R&B;']['sim'].min()
jz_same_min=same[same['genre1']=='Jazz']['sim'].min()

pop_rb_min=diff.loc[(diff['genre0']=='Pop/Rock')&(diff['genre1']=='R&B;')]['sim'].min()
pop_jz_min=diff.loc[(diff['genre0']=='Pop/Rock')&(diff['genre1']=='Jazz')]['sim'].min()
rb_jz_min=diff.loc[(diff['genre0']=='R&B;')&(diff['genre1']=='Jazz')]['sim'].min()

pop_same_max=same[same['genre0']=='Pop/Rock']['sim'].max()
rb_same_max=same[same['genre1']=='R&B;']['sim'].max()
jz_same_max=same[same['genre1']=='Jazz']['sim'].max()

pop_rb_max=diff.loc[(diff['genre0']=='Pop/Rock')&(diff['genre1']=='R&B;')]['sim'].max()
pop_jz_max=diff.loc[(diff['genre0']=='Pop/Rock')&(diff['genre1']=='Jazz')]['sim'].max()
rb_jz_max=diff.loc[(diff['genre0']=='R&B;')&(diff['genre1']=='Jazz')]['sim'].max()

score=defaultdict(list)
score['pop_same']=[pop_same_mean,pop_same_min,pop_same_max]
score['rb_same']=[rb_same_mean,rb_same_min,rb_same_max]
score['jz_same']=[jz_same_mean,jz_same_min,jz_same_max]
score['pop_rb']=[pop_rb_mean,pop_rb_min,pop_rb_max]
score['pop_jz']=[pop_jz_mean,pop_jz_min,pop_jz_max]
score['rb_jz']=[rb_jz_mean,rb_jz_min,rb_jz_max]
score=pd.DataFrame(score)
score.to_csv('流派差异.csv')