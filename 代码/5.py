import pandas as pd
import numpy as np
from collections import defaultdict

genres = ['Avant-Garde', 'Blues', "Children's", 'Classical',
          'Comedy/Spoken', 'Country', 'Easy Listening', 'Electronic',
          'Folk', 'International', 'Jazz', 'Latin', 'New Age', 'Pop/Rock',
          'R&B;', 'Reggae', 'Religious', 'Stage & Screen', 'Vocal']
music = pd.read_csv('/content/drive/MyDrive/dataset/2021D/full_music_data.csv')
influence = pd.read_csv('/content/drive/MyDrive/dataset/2021D/influence_data.csv')


def make_label(x):
    x = x[1:-1]
    x = x.split(',')
    if len(x) == 1:
        return 0
    else:
        return 1


music['label'] = music['artists_id'].apply(make_label)
music = music[music['label'] == 0]


def make_artists(x):
    x = x[1:-1]
    x = x.split(',')
    return x[0]


music['artists_id'] = music['artists_id'].apply(make_artists)


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


follower_name = \
influence.loc[(influence['influencer_main_genre'] == 'Pop/Rock') & (influence['follower_main_genre'] == 'Pop/Rock')][
    'follower_id'].unique().tolist()
names = []
for name in follower_name:
    info = music[music['artists_id'] == str(name)]
    if info.shape[0] > 10:
        names.append(name)

year_dict = {}
for year in range(1960, 2021, 10):
    for name in names:
        inf_names = \
        influence.loc[(influence['follower_id'] == name) & (influence['influencer_main_genre'] == 'Pop/Rock')][
            'influencer_id'].unique().tolist()
        for inf_name in inf_names:
            inf_music = music[music['artists_id'] == str(inf_name)]
            inf_music = inf_music[inf_music['year'].isin(list(range(year, year + 11)))]

            if inf_music.shape[0] <= 10:
                continue
            if inf_music.shape[0] > 100:
                index = list(range(0, inf_music.shape[0]))
                index = np.random.choice(index, 100, replace=False)
                inf_music = inf_music.iloc[index]

            follow_music = music[music['artists_id'] == str(name)]
            follow_music = follow_music[follow_music['year'].isin(list(range(year, year + 11)))]
            if follow_music.shape[0] <= 10:
                continue
            if follow_music.shape[0] > 100:
                index = list(range(0, follow_music.shape[0]))
                index = np.random.choice(index, 100, replace=False)
                inf_music = follow_music.iloc[index]

            trend, t_sum = count_trend(follow_music, inf_music)
            L = max(0, t_sum) + max(0, trend - 0.003)
            L_dict[inf_name] += L