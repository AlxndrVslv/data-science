#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import math

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse


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


# Видно, что таких всего 7. Заменим их на NaN:

# In[5]:


movies_df['Year'] = movies_df['Year'].replace('NULL', pd.NA)


# Приведем ИД фильма к числовому типу:

# In[6]:


movies_df = movies_df.astype({'Movie_Id': 'int32'})
movies_df.head()


# Файл с рейтингами очень большой, поэтому для тестов сначала возьмем 1 млн. строк:

# In[7]:


ratings_df = pd.read_csv(
    'total_ratings.csv',
    header = None,
    names = ['Movie_Id', 'Cust_Id', 'Rating', 'Date'],
    dtype = {
        'Movie_Id': 'int16',
        'Cust_Id': 'int32',
        'Rating': 'int8'
    },
    parse_dates = ['Date'],
    nrows = 10**6
)

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


threshold_dt = datetime(2004, 1, 1)

while threshold_dt < datetime(2006, 1, 1):
    train_cnt = ratings_df[ratings_df['Date'] < threshold_dt]['Rating'].count()
    test_cnt = ratings_df[ratings_df['Date'] >= threshold_dt]['Rating'].count()
    print(f'На дату {threshold_dt} доля тестовой выборки равна {test_cnt / (train_cnt + test_cnt): .2f}')
    threshold_dt += relativedelta(months = 1)


# Видно, что такое соотношение достигается, если выбрать порогом 01.08.2005.

# In[15]:


df_train = df[df['Date'] < datetime(2005, 8, 1)]
df_test = df[df['Date'] >= datetime(2005, 8, 1)]

print(df_train.shape, df_test.shape)


# ## 3. Разработка моделей рекомендаций:

# В справочнике фильмов для контент-ориентированной рекомендации есть всего 2 признака: год выхода фильма и название.<br/>
# Вряд ли на основании только этих признаков можно создать качественную модель.

# conda activate netology_env<br/>
# jupyter nbconvert --to script HW5.ipynb

# In[47]:


class Hybrid:

    def __init__(self, df, user_id):
        self.df = df
        self.user_id = user_id

    ############################################################################################################################

    def _get_basic_scores(self, n_recommendations = 10, min_votes_for_recommend = 50):

        high_rated_movies = self.df[self.df['Rating'] >= 4].groupby('Movie_Id')['Rating'].agg(['mean', 'count'])
        high_rated_movies = high_rated_movies[high_rated_movies['count'] >= min_votes_for_recommend]
        recommendations = high_rated_movies.sort_values(by = ['mean', 'count'], ascending = False).head(n_recommendations).reset_index()
        recommendations = recommendations[['Movie_Id', 'mean']]
        recommendations.columns = ['Movie_Id', 'Score']
        return recommendations

    def get_basic_recommendations(self, n_recommendations = 10, min_votes_for_recommend = 50):
        '''
        n_recommendations: кол-во рекомендаций
        min_votes_for_recommend: минимальное кол-во голосов для отбора
        :return: Базовые рекомендации на основе самых популярных фильмов
        '''

        movie_ids = self._get_basic_scores(n_recommendations = n_recommendations, min_votes_for_recommend = min_votes_for_recommend)['Movie_Id'].tolist()

        return movies_df[movies_df['Movie_Id'].isin(movie_ids)]

    ############################################################################################################################

    def _create_tfidf_matrix(self):
        df_tmp = movies_df.copy()
        df_tmp['features'] = df_tmp['Year'].fillna('') + ' ' + df_tmp['Name']

        tfidf = TfidfVectorizer(stop_words = 'english')
        tfidf_matrix = tfidf.fit_transform(df_tmp['features'])

        return tfidf_matrix


    def _get_content_scores(self, n_recommendations = 10):

        rated_movie_ids = self.df[(self.df['Cust_Id'] == self.user_id) & (self.df['Rating'] >= 4)]['Movie_Id'].to_list()

        if not rated_movie_ids:
            return self.get_basic_recommendations(n_recommendations)

        tfidf_matrix = self._create_tfidf_matrix()
        cosine_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        rated_films_cnt = len(rated_movie_ids)
        similars_cnt_per_movie = math.ceil(n_recommendations * 1./ rated_films_cnt)

        best_movies = {}
        for rated_movie_id in rated_movie_ids:
            rated_movie_ind = movies_df[movies_df['Movie_Id'] == rated_movie_id].index[0]
            similar_movie_inds = cosine_similarity_matrix[rated_movie_ind].argsort()[::-1][1: similars_cnt_per_movie + 1]
            for similar_movie_ind in similar_movie_inds:
                similar_movie_value = cosine_similarity_matrix[rated_movie_ind][similar_movie_ind]
                similar_movie_id = movies_df[movies_df.index == similar_movie_ind]['Movie_Id'].item()
                best_movies[similar_movie_id] = similar_movie_value

        n_best_movies = dict(sorted(best_movies.items(), key = lambda x: x[1], reverse = True)[:n_recommendations])

        return pd.DataFrame(n_best_movies.items(), columns = ['Movie_Id', 'Score'])

    def get_content_recommendations(self, n_recommendations = 10):
        '''
        n_recommendations: кол-во рекомендаций
        :return: Рекомендации на основе контента
        '''

        movie_ids = self._get_content_scores(n_recommendations = n_recommendations)['Movie_Id'].tolist()

        return movies_df[movies_df['Movie_Id'].isin(movie_ids)]

    ############################################################################################################################

    def _create_model(self, df, model):

        reader = Reader(rating_scale = (1, 5))
        data = Dataset.load_from_df(df = df, reader = reader)

        trainset, testset = train_test_split(data = data, test_size = 0.2, random_state = 42)
        model.fit(trainset)

        predict = model.test(testset)
        model_accuracy = rmse(predict, verbose = False)

        return model, model_accuracy


    def _get_collaborative_scores(self, n_recommendations = 10):

        rated_movie_ids = np.unique(self.df[self.df['Cust_Id'] == self.user_id]['Movie_Id'].tolist())
        unrated_movie_ids = movies_df[~movies_df['Movie_Id'].isin(rated_movie_ids)]['Movie_Id'].to_list()

        df_colab = self.df[['Cust_Id', 'Movie_Id', 'Rating']]
        df_colab.columns = ['user_id', 'item_id', 'rating']

        model, model_accuracy = self._create_model(
            df = df_colab,
            model = SVD(n_factors = 100, n_epochs = 30, lr_all = 0.005, reg_all = 0.02, random_state = 42)
        )

        best_movies = {}
        for unrated_movie_id in unrated_movie_ids:
            rate_predicted = model.predict(uid = self.user_id, iid = unrated_movie_id).est
            best_movies[unrated_movie_id] = rate_predicted

        n_best_movies = dict(sorted(best_movies.items(), key = lambda x: x[1], reverse = True)[:n_recommendations])

        return pd.DataFrame(n_best_movies.items(), columns = ['Movie_Id', 'Score'])


    def get_collaborative_recommendations(self, n_recommendations = 10):
        '''
        n_recommendations: кол-во рекомендаций
        :return: Коллаборативные рекомендации
        '''

        movie_ids = self._get_collaborative_scores(n_recommendations = n_recommendations)['Movie_Id']

        return movies_df[movies_df['Movie_Id'].isin(movie_ids)]

    ############################################################################################################################

    @staticmethod
    def get_scaled_scores(df, column):
        df_copy = df.copy()
        scaler = StandardScaler()

        X = df[column].values.reshape(-1, 1)
        X_scaled = scaler.fit_transform(X)
        df_copy['Scaled'] = X_scaled

        return df_copy


    def get_hybrid_recommendations(self, n_recommendations = 10, w_content = 0.3):

       content_df = self._get_content_scores(n_recommendations)
       collab_df = self._get_collaborative_scores(n_recommendations)

       content_df = self.get_scaled_scores(content_df, 'Score')
       collab_df = self.get_scaled_scores(collab_df, 'Score')

       content_df = content_df[['Movie_Id', 'Scaled']]
       collab_df = collab_df[['Movie_Id', 'Scaled']]

       union_df = content_df.merge(collab_df, how = 'outer', on = 'Movie_Id')

       union_df.fillna(0, inplace = True)
       union_df.columns = ['Movie_Id', 'Score_content', 'Score_collab']

       union_df['Final_score'] = w_content * union_df['Score_content'] + (1 - w_content) * union_df['Score_collab']

       best_movie_ids = union_df.sort_values('Final_score', ascending = False).head(n_recommendations)['Movie_Id'].tolist()

       return movies_df[movies_df['Movie_Id'].isin(best_movie_ids)]


# In[55]:


recommend = Hybrid(
    df = df_train,
    user_id = 1461435 #644003
)


# In[56]:


get_ipython().run_cell_magic('time', '', 'br = recommend.get_basic_recommendations(n_recommendations = 5, min_votes_for_recommend = 10000)\nbr\n')


# In[57]:


get_ipython().run_cell_magic('time', '', 'cnr = recommend.get_content_recommendations(n_recommendations = 5)\ncnr\n')


# In[58]:


get_ipython().run_cell_magic('time', '', 'cbr = recommend.get_collaborative_recommendations(n_recommendations = 5)\ncbr\n')


# In[59]:


get_ipython().run_cell_magic('time', '', 'hr = recommend.get_hybrid_recommendations(n_recommendations = 5)\nhr\n')


# In[37]:


real_ratings = {}

for usr_id, itm_id, rate in df_test[['Cust_Id', 'Movie_Id', 'Rating']].astype(int).values:
    if usr_id not in real_ratings:
        real_ratings[usr_id] = {}
    real_ratings[usr_id][itm_id] = rate


# In[43]:


recommend = Hybrid(df_train, 1)
recommend._get_content_scores()


# In[60]:


get_ipython().run_cell_magic('time', '', "recommendations = {}\n\ni = 0\n\nfor usr_id in real_ratings:\n    recommend = Hybrid(df_test, usr_id)\n    recommendations[usr_id] = recommend._get_content_scores()['Movie_Id'].tolist()\n    i += 1\n    if i == 100: break\n\n# 37.9 - 100\n# 379 - 1000\n")

