#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict

import os
from scipy.sparse import save_npz, load_npz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split as sklearn_split


from surprise import Dataset, Reader, SVD
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split as surprice_split
from surprise.prediction_algorithms import predictions

# ## 1. Исследовательский анализ данных (Exploratory Data Analysis - EDA):

# Для рекомендательной системы будем использовать дата-сет с оценками фильмов Netflix (Netflix Prize Data Set).

# Т.к. файл с названиями фильмов в качестве разделителя использует запятую и в названии фильмов может быть запятая, то с загрузкой через pd.read_csv могут возникнуть проблемы.<br/>
# Но в файле название фильма идет на последней правой позиции. Поэтому прочитаем файл построчно, и последнюю позицию выделим под называние фильма.

# In[2]:


with open('movie_titles.txt', 'r', encoding = 'ISO-8859-1') as f:
    lines = f.readlines()

data = []
for line in lines:
    parts = line.strip().split(',')
    movie_id = parts[0]
    year = parts[1]
    name = ','.join(parts[2:])
    data.append([movie_id, year, name])

movies_df = pd.DataFrame(data, columns = ['Movie_Id', 'Year', 'Name'])


# Выведем уникальные значения года выхода фильма:

# In[3]:


print(sorted(movies_df['Year'].unique()))


# Видно, что есть не значимые строковые значения 'NULL'. Посмотрим, сколько таких значений:

# In[4]:


movies_df[movies_df['Year'] == 'NULL']


# Видно, что таких всего 7. Заменим их на 1900 год:

# In[5]:


movies_df['Year'] = movies_df['Year'].replace('NULL', '1900')


# Приведем ИД фильма к числовому типу:

# In[6]:


movies_df = movies_df.astype({'Movie_Id': 'int32'})
movies_df.head()


# Файл с рейтингами очень большой, поэтому для тестов сначала возьмем 1 млн. строк:

# In[7]:


# ratings_df = pd.read_csv(
#     'total_ratings.csv',
#     header = None,
#     names = ['Movie_Id', 'Cust_Id', 'Rating', 'Date'],
#     dtype = {
#         'Movie_Id': 'int16',
#         'Cust_Id': 'int32',
#         'Rating': 'int8'
#     },
#     parse_dates = ['Date'],
#     nrows = 10**6
# )

# Шаг 1: читаем рейтинги по chunks, собираем по 5000 случайных пользователям
all_users = set()

for chunk in pd.read_csv('total_ratings.csv', header=None,
                         names=['Movie_Id', 'Cust_Id', 'Rating', 'Date'],
                         dtype={'Movie_Id': 'int32', 'Cust_Id': 'int32', 'Rating': 'int8'},
                         chunksize=1_000_000):
    all_users.update(chunk['Cust_Id'].unique())

# Шаг 2: выбираем 5000 случайных пользователей
np.random.seed(42)
selected_users = set(np.random.choice(list(all_users), 5000, replace=False))

# Шаг 3: читаем снова, оставляя только их
chunks = []
for chunk in pd.read_csv('total_ratings.csv', header=None,
                         names=['Movie_Id', 'Cust_Id', 'Rating', 'Date'],
                         dtype={'Movie_Id': 'int32', 'Cust_Id': 'int32', 'Rating': 'int8'},
                         parse_dates=['Date'],
                         chunksize=1_000_000):
    filtered = chunk[chunk['Cust_Id'].isin(selected_users)]
    if len(filtered) > 0:
        chunks.append(filtered)

ratings_df = pd.concat(chunks, ignore_index=True)
print(f"Оценок: {len(ratings_df)}, Фильмов: {ratings_df['Movie_Id'].nunique()}")

ratings_df.head()


# Сформируем объединенный датасет из справочника фильмов и их рейтингов:

# In[8]:


df = movies_df.merge(ratings_df, how = 'inner', on = 'Movie_Id')
df.head()


# In[9]:


df.info()


# Видно, что в итоговом датасете отсутствуют пропуски.

# In[10]:


df.describe()


# Видим, что диапазон рейтингов лежит в пределах от 1 до 5. Выбросов нет.<br/>
# Даты установки рейтингов в пределах 1999 - 2006 годов.

# Проанализируем распределение рейтингов в наборе данных:

# In[11]:


fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (24, 6))
fig.suptitle('Распределение оценок фильмов:')
fig.tight_layout(w_pad = 4.0)

order = sorted(df['Year'].unique())
sns.countplot(data = df, x = 'Rating', hue = 'Rating', palette = 'pastel', legend = False, ax = axes[0])
sns.histplot(data = df, x = 'Date', hue = 'Rating', kde = True, palette = 'pastel', ax = axes[1])
sns.countplot(data = df, x = 'Year', color = 'g', order = order, ax = axes[2])
axes[2].set_xticks(order)
axes[2].set_xticklabels(labels = axes[2].get_xticklabels(), rotation = 90)

plt.show()


# <ul>Из графиков ввидно:
# <li>Самая популярная оценка равна 4;</li>
# <li>С 2002 года постепенно растет кол-во оценок и достигает пика в 2005;</li>
# <li>Самыми популярными являются фильмы 2000-2003 годов выпуска;</li>
# </u>

# Найдем самые полулярные фильмы с оценками 4 или 5 баллов:

# In[12]:


df[df['Rating'] >= 4].groupby(['Movie_Id', 'Name'])['Rating'].count().sort_values(ascending = False).head(10)


# Найдем самых активных пользователей:

# In[13]:


ratings_df.groupby('Cust_Id')['Rating'].count().sort_values(ascending = False).head(10)


# ## 2. Подготовка данных (Data Preprocessing):

# Сформируем тренировочную и тестовую выборки, основываясь на датах рейтингов.

# Необходимо найти такую пороговую дату, чтобы соотношение тестовой и тренировочной выборки приблизительно было равно 20/80.

# In[14]:


# threshold_dt = datetime(2004, 1, 1)
#
# while threshold_dt < datetime(2006, 1, 1):
#     train_cnt = ratings_df[ratings_df['Date'] < threshold_dt]['Rating'].count()
#     test_cnt = ratings_df[ratings_df['Date'] >= threshold_dt]['Rating'].count()
#     print(f'На дату {threshold_dt} доля тестовой выборки равна {test_cnt / (train_cnt + test_cnt): .2f}')
#     threshold_dt += relativedelta(months = 1)


# Видно, что такое соотношение достигается, если выбрать порогом 01.08.2005.

# In[15]:


# df_train = df[df['Date'] < datetime(2005, 8, 1)]
# df_test = df[df['Date'] >= datetime(2005, 8, 1)]

df_train, df_test = sklearn_split(df, test_size = 0.2, random_state = 42)

print(df_train.shape, df_test.shape)


# ## 3. Разработка моделей рекомендаций:

# В справочнике фильмов для контент-ориентированной рекомендации есть всего 2 признака: год выхода фильма и название.<br/>
# Вряд ли на основании только этих признаков можно создать качественную модель.

# conda activate netology_env<br/>
# jupyter nbconvert --to script HW5.ipynb

# In[33]:


class Hybrid:

    def __init__(self, movies_list, df_train, df_test, text_column = 'Feature', precompute_path = 'precomputed/'):

        self.movies_df = movies_list
        self.df_train = df_train
        self.df_test = df_test

        self.precompute_path = precompute_path
        os.makedirs(precompute_path, exist_ok = True)

        self._create_tfidf_matrix(text_column)
        self._create_cosine_sim()
        self._init_top_similar()

    ############################################################################################################################
    # Вычисление матрицы схожести

    def _create_tfidf_matrix(self, text_column):

        tfidf_path = self.precompute_path + 'tfidf_matrix.npz'

        if os.path.exists(tfidf_path):
            print('Загрузка TF-IDF матрицы...')
            self.tfidf_matrix = load_npz(tfidf_path)
        else:
            print('Вычисление TF-IDF матрицы...')
            df_tmp = self.movies_df.copy()
            df_tmp[text_column] = df_tmp['Year'].fillna('') + ' ' + df_tmp['Name']
            tfidf = TfidfVectorizer(stop_words = 'english')
            self.tfidf_matrix = tfidf.fit_transform(df_tmp[text_column].fillna(''))
            save_npz(tfidf_path, self.tfidf_matrix)

        self.movie_ids = self.movies_df['Movie_Id'].tolist()
        # self.movie_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

    def _create_cosine_sim(self):

        cosine_sim_path = self.precompute_path + 'cosine_sim.npy'

        if os.path.exists(cosine_sim_path):
            print('Загрузка матрицы схожести...')
            self.cosine_sim = np.load(cosine_sim_path)
        else:
            print('Вычисление матрицы схожести... (может занять время)')
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            np.save(cosine_sim_path, self.cosine_sim)
            print(f'Матрица сохранена: {cosine_sim_path}')

    ############################################################################################################################

    def _init_top_similar(self, top_n = 50):

        top_similar_path = self.precompute_path + f'top_{top_n}_similar.pkl'

        if os.path.exists(top_similar_path):
            print(f'Загрузка top_{top_n} похожих фильмов...')
            import pickle
            with open(top_similar_path, 'rb') as f:
                self.top_similar = pickle.load(f)
        else:
            print(f'Вычисление top_{top_n} похожих фильмов...')
            self.top_similar = {}

            for idx, movie_id in enumerate(self.movie_ids):

                similarities = self.cosine_sim[idx]
                top_indices = similarities.argsort()[-top_n - 1:-1][::-1]
                self.top_similar[movie_id] = [self.movie_ids[i] for i in top_indices]

                if (idx + 1) % 5000 == 0:
                    print(f'Обработано {idx + 1} из {len(self.movie_ids)}')

            import pickle
            with open(top_similar_path, 'wb') as f:
                pickle.dump(self.top_similar, f)
        print('Готово')

    ############################################################################################################################

    def _get_popular_movies(self, df, k = 10):

        popular_movie_ids = df[df['Rating'] >= 4] \
            .groupby('Movie_Id')['Rating'].count() \
            .sort_values(ascending = False).head(k) \
            .index.tolist()

        return popular_movie_ids

    ############################################################################################################################

    def _create_position_scores(self, lst):
        l = len(lst) - 1
        return [(elem, (l - i) / l) for i, elem in enumerate(lst)]

    ############################################################################################################################

    def get_content_recommends(self, df, k = 10):

        top_n = defaultdict(list)

        for user_id in df['Cust_Id'].unique():

            # Если у пользователя больше 5 оценок, то контентную модель не применяем. Ее применяем только для холодного старта
            if df[df['Cust_Id'] == user_id]['Rating'].count() > 5:
                top_n[user_id] = [(0, 0.0),]
                continue

            user_liked_movies = df[(df['Cust_Id'] == user_id) & (df['Rating'] >= 4)]['Movie_Id'].tolist()

            if not user_liked_movies:
                top_n[user_id] = self._create_position_scores(self._get_popular_movies(df, k))
                continue

            candidates = {}

            for liked_movie in user_liked_movies:
                for similar_movie in self.top_similar.get(liked_movie, []):
                    candidates[similar_movie] = candidates.get(similar_movie, 0) + 1

            for liked_movie in user_liked_movies:
                candidates.pop(liked_movie, None)

            sorted_candidates = sorted(candidates.items(), key = lambda x: x[1], reverse = True)[:k]
            sorted_candidates = [mid for mid, _ in sorted_candidates]
            positioned_scores = self._create_position_scores(sorted_candidates)

            top_n[user_id] = positioned_scores

        self.content_recs = top_n

    ############################################################################################################################

    def _create_model(self, df):

        df_copy = df[['Cust_Id', 'Movie_Id', 'Rating']].copy()
        df_copy.columns = ['user_id', 'item_id', 'rating']

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_copy, reader)

        trainset = data.build_full_trainset()

        model = SVD(n_factors=100, n_epochs=30, lr_all=0.005, reg_all=0.02)
        model.fit(trainset)

        return model

    def _create_testset(self):

        df_copy = self.df_test[['Cust_Id', 'Movie_Id', 'Rating']].copy()
        df_copy.columns = ['user_id', 'item_id', 'rating']

        testset = list(
            zip(
                df_copy['user_id'].values,
                df_copy['item_id'].values,
                df_copy['rating'].values
            )
        )

        return testset

    def get_collaborative_recommends(self, df, k = 10, rmse_only = False):

        top_n = defaultdict(list)

        model = self._create_model(df)

        testset = self._create_testset()
        predictions = model.test(testset)

        if rmse_only:
            return rmse(predictions, verbose = True)

        for uid, iid, _, est, _ in predictions:
            top_n[uid].append((iid, est))

        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            min = np.min([r for mid, r in user_ratings])
            max = np.max([r for mid, r in user_ratings])
            user_ratings_scaled = [(iid, (r - min) / (max - min) if (max > min) else 0.5) for iid, r in user_ratings]

            top_n[uid] = user_ratings_scaled[:k]

        self.collab_recs = top_n



    def get_rmse(self):
        return self.get_collaborative_recommends(df = self.df_train, rmse_only = True)

    ############################################################################################################################

    def _dict_to_df(self, dict_recs):

        content_list = []

        for uid, user_ratings in dict_recs.items():
            for iid, rating in user_ratings:
                content_list.append([uid, iid, rating])

        return pd.DataFrame(content_list, columns = ['uid', 'iid', 'rating'])

    ############################################################################################################################

    def get_hybrid_recommends(self, df, k = 10, w_content = 0.3):

        if not hasattr(self, 'content_recs'):
            print('Вычисление контентных рекомендаций...')
            self.get_content_recommends(df, k * 2)
            print('Готово!')

        if not hasattr(self, 'collab_recs'):
            print('Вычисление коллаборативных рекомендаций...')
            self.get_collaborative_recommends(df, k * 2)
            print('Готово!')

        print('Вычисление гибридных рекомендаций...')

        content_df = self._dict_to_df(self.content_recs)
        collab_df = self._dict_to_df(self.collab_recs)

        content_df['w_content_rating'] = content_df['rating'] * w_content
        collab_df['w_collab_rating'] = collab_df['rating'] * (1 - w_content)

        union_df = content_df[['uid', 'iid', 'w_content_rating']].merge(right = collab_df[['uid', 'iid', 'w_collab_rating']], how = 'outer', on = ['uid', 'iid']).fillna(0)
        union_df['total_rating'] = union_df['w_content_rating'] + union_df['w_collab_rating']
        union_df = union_df[['uid', 'iid', 'total_rating']]

        hybrid_recs = {}
        for uid, group in union_df.groupby('uid'):
            sorted_group = group.sort_values(by = 'total_rating', ascending = False)
            hybrid_recs[uid] = list(zip(sorted_group['iid'], sorted_group['total_rating']))[:k]

        self.hybrid_recs = hybrid_recs


    def get_quality_metrics(self, k = 10, w_content = 0.3, threshold = 4):

        if not hasattr(self, 'content_recs'): self.get_content_recommends(df = self.df_train, k = k)
        if not hasattr(self, 'collab_recs'): self.get_collaborative_recommends(df = self.df_train, k = k)
        if not hasattr(self, 'hybrid_recs'): self.get_hybrid_recommends(df = self.df_train, k = k, w_content = w_content)

        user_real_ratings = defaultdict(dict)

        for user_id, movie_id, rating in zip(
            self.df_test['Cust_Id'],
            self.df_test['Movie_Id'],
            self.df_test['Rating']
        ):
            user_real_ratings[user_id][movie_id] = rating

        recommendations = self.content_recs, self.collab_recs, self.hybrid_recs

        model_types = ['content', 'collaborative', 'hybrid']
        models_dict = dict(zip(model_types, recommendations))

        result = defaultdict(dict)

        for model_type, recommendation in models_dict.items():

            precisions, recalls, aps = [], [], []

            for user_id, rec_list in recommendation.items():
                if user_id not in user_real_ratings:
                    continue

                real_ratings = user_real_ratings[user_id]

                # Precision@k
                relevant_precisions_count = 0
                for movie_id, rating in rec_list[:k]:
                    if movie_id in real_ratings and real_ratings[movie_id] >= threshold:
                        relevant_precisions_count += 1

                l = len(rec_list[:k])
                precisions.append(relevant_precisions_count / l if l else 0)

                # Recall@k
                relevant_recalls_count = 0
                for movie_id in real_ratings:
                    if movie_id in [m_id for m_id, r in rec_list[:k]] and real_ratings[movie_id] >= threshold:
                        relevant_recalls_count += 1

                l = len([m_id for m_id, r in real_ratings.items() if r >= threshold])
                recalls.append(relevant_recalls_count / l if l else 0)

                # MAP@k
                hits = 0
                ap_sum = 0.0

                for i, (movie_id, r) in enumerate(rec_list[:k], start=1):
                    if movie_id in real_ratings and real_ratings[movie_id] >= threshold:
                        hits += 1
                        ap_sum += hits / i

                n_relevant = sum(1 for r in real_ratings.values() if r >= threshold)
                aps.append(ap_sum / min(n_relevant, k) if n_relevant else 0.0)

            result[model_type]['precisions'] = round(sum(precisions) / len(precisions), 5) if precisions else 0.0
            result[model_type]['recalls'] = round(sum(recalls) / len(recalls), 5) if recalls else 0.0
            result[model_type]['maps'] = round(sum(aps) / len(aps), 5) if aps else 0.0


        return result


# In[34]:


# recommend = Hybrid(movies_list = movies_df, df_train = df_train.head(50000), df_test = df_test.head(12000))
recommend = Hybrid(movies_list = movies_df, df_train = df_train, df_test = df_test)

# recommend.get_hybrid_recommends(df_train, k = 10)

# precisions = recommend.get_quality_metrics()

rmse = recommend.get_rmse()
print(f'Качество модели коллаборативной фильтрации = {rmse:.5f}')

# In[28]:


# user_id = 588844
#
# print(f'Для пользователя {user_id} рекомендованные фильмы (hybrid):')
# movies_df[movies_df['Movie_Id'].isin([movie_id for movie_id in user_collab_recs[user_id]])]


# In[31]:


user_content_recs


# In[32]:


user_collab_recs

