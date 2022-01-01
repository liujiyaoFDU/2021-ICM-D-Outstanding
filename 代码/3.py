import numpy as np
import pandas as pd
influencer_name=influence['influencer_id'].unique().tolist()
year_dict={}
genre_dict={}
for id in influencer_name:
  genres=influence[influence['influencer_id']==id]['influencer_main_genre'].unique().tolist()
  year=influence[influence['influencer_id']==id]['influencer_active_start'].unique().tolist()
  if len(genres)!=1:
    print('False')
  genre_dict[id]=genres[0]
  year_dict[id]=year[0]

follower_name=influence['follower_id'].unique().tolist()
for id in follower_name:
  genres=influence[influence['follower_id']==id]['follower_main_genre'].unique().tolist()
  year=influence[influence['follower_id']==id]['follower_active_start'].unique().tolist()
  if len(genres)!=1:
    print('False')
  genre_dict[id]=genres[0]
  year_dict[id]=year[0]
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
name_list1 = list(name_list1)
name_list2 = list(name_list2)
name_list3 = list(name_list3)
name_list1.extend(name_list2)
name_list1.extend(name_list3)

feature_columns=['danceability','energy','valence','tempo','loudness','mode','key','acousticness','instrumentalness','liveness','speechiness','duration_ms']
train_columns=[]
for i in range(len(feature_columns)-1):
  for j in range(i+1,len(feature)):
    feature1=feature_columns[i]
    feature2=feature_columns[j]
    train_columns.append([feature1,feature2])

for i in range(len(feature_columns)-2):
  for j in range(i+1,len(feature_columns)-1):
    for k in range(j+1,len(feature_columns)):
      feature1=feature_columns[i]
      feature2=feature_columns[j]
      feature3=feature_columns[k]
      train_columns.append([feature1,feature2,feature3])

for i in range(len(feature_columns)-3):
  for j in range(i+1,len(feature_columns)-2):
    for k in range(j+1,len(feature_columns)-1):
      for f in range(k+1,len(feature_columns)):
        feature1=feature_columns[i]
        feature2=feature_columns[j]
        feature3=feature_columns[k]
        feature4=feature_columns[f]
        train_columns.append([feature1,feature2,feature3,feature4])


for i in range(len(feature_columns)-4):
  for j in range(i+1,len(feature_columns)-3):
    for k in range(j+1,len(feature_columns)-2):
      for f in range(k+1,len(feature_columns)-1):
        for g in range(f+1,len(feature_columns)):
          feature1=feature_columns[i]
          feature2=feature_columns[j]
          feature3=feature_columns[k]
          feature4=feature_columns[f]
          feature5=feature_columns[g]
          train_columns.append([feature1,feature2,feature3,feature4,feature5])
def make_label(x):
  if x['genre0']==x['genre1']:
    return 0
  else:
    return 1

def make_gener(x):
  if x not in genre_dict.keys():
    return 0
  return genre_dict[x]

artist=pd.read_csv('/content/drive/MyDrive/dataset/2021D/data_by_artist.csv')
feature_columns=['artist_id','danceability','energy','valence','tempo','loudness','mode','key','acousticness','instrumentalness','liveness','speechiness','duration_ms']
artist_feature=artist[feature_columns]
artist_feature['genre']=artist_feature['artist_id'].apply(make_gener)
artist_feature=artist_feature[artist_feature['genre']!=0]
nor_columns=['tempo','key','duration_ms']
for column in nor_columns:
  artist_feature.loc[:,column]=artist_feature[column]/artist_feature[column].max()
artist_feature.loc[:,'loudness']=artist_feature['loudness']/25
genres=['Avant-Garde','Blues',"Children's",'Classical',
    'Comedy/Spoken','Country','Easy Listening','Electronic',
    'Folk','International','Jazz','Latin','New Age','Pop/Rock',
    'R&B;','Reggae','Religious','Stage & Screen','Vocal']
feature_columns=['danceability','energy','valence','tempo','loudness','mode','key','acousticness','instrumentalness','liveness','speechiness','duration_ms']
train_data=np.zeros((len(genres),12))
for i,genre in enumerate(genres):
  data=artist_feature[artist_feature['genre']==genre][feature_columns].mean()
  train_data[i]=data

train_data=pd.DataFrame(train_data)
train_data.columns=feature_columns
train_data['genre']=genres


def make_local_distance_sort(x1, x2, column, dis_type):
    x1 = x1[column]
    x2 = x2[column]
    sim = make_global_distance(x1, x2, dis_type)
    return sim


from tqdm import tqdm

dis_type = 'both'
feature_columns = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness', 'duration_ms']
music_global_sim = defaultdict(list)
music_all_sim = defaultdict(list)
target = [0.5 / 1.5] * (len(genres) - 1)
target = np.array(target)
print(target.shape)
score_dict = defaultdict(list)

for i in range(train_data.shape[0]):
    info1 = train_data.iloc[i]
    genre1 = info1['genre']
    x1 = info1[feature_columns]
    genre1 = genre1.replace('/', ' ')
    genre1 = genre1.replace('&', ' ')
    genre1 = genre1.replace(';', ' ')

    for j in range(train_data.shape[0]):
        if j == i:
            continue
        info2 = train_data.iloc[j]
        genre2 = info2['genre']
        x2 = info2[feature_columns]
        score_dict['genre'].append(genre2)

        for column in train_columns:
            name = ' '.join(column)
            sim = make_local_distance_sort(x1, x2, column, dis_type)
            score_dict[name].append(sim)

    score_dict = pd.DataFrame(score_dict)
    score_dict = score_dict.to_csv('{}.csv'.format(genre1))



