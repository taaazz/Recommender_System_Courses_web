import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Concatenate
from tensorflow.keras.models import Model

# Define the RecommenderNet class
class RecommenderNet(Model):
    def __init__(self, num_users, num_courses, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_courses = num_courses
        self.embedding_size = embedding_size
        self.user_embedding = Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer='l2')
        self.course_embedding = Embedding(num_courses, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer='l2')
        self.user_bias = Embedding(num_users, 1)
        self.course_bias = Embedding(num_courses, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        course_vector = self.course_embedding(inputs[:, 1])
        user_bias = self.user_bias(inputs[:, 0])
        course_bias = self.course_bias(inputs[:, 1])

        dot_user_course = tf.tensordot(user_vector, course_vector, 2)

        # Add all the components (dot product + bias terms)
        x = dot_user_course + user_bias + course_bias
        return tf.nn.sigmoid(x)

    def get_config(self):
        return {
            'num_users': self.num_users,
            'num_courses': self.num_courses,
            'embedding_size': self.embedding_size,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load the required data and model
final_df = pd.read_csv('data_prep.csv')
final_rating_df = pd.read_csv('final_rating_df.csv')

# Adjust the num_users and num_courses according to your data
num_users = final_df['user_id'].nunique()
num_courses = final_df['course_id'].nunique()

# Load the model
model = load_model('recommender_model.h5', custom_objects={'RecommenderNet': RecommenderNet})

# Define dictionaries to decode user and course ids (adjust if needed)
users_decoded = dict(final_df[['user_id', 'user_name']].values)
courses_decoded = dict(final_df[['course_id', 'name']].values)

# Define the recommendation function
def get_recommendations(user_id):
    reviewed_course_by_user = final_df[final_df.user_id == user_id]
    courses_not_reviewed = final_df[~(final_df.name
                                      .isin(reviewed_course_by_user.name.values)
                                      )]['course_id']
    courses_not_reviewed = list(
        set(courses_not_reviewed)
        .intersection(set(courses_decoded.keys()))
    )

    courses_not_reviewed = [[x] for x in courses_not_reviewed]
    user_courses_array = np.hstack(
        ([[user_id]] * len(courses_not_reviewed), courses_not_reviewed)
    )

    ratings = model.predict(user_courses_array, verbose=0).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]

    top_courses_user = reviewed_course_by_user.sort_values(
        by='rating', ascending=False
    ).head(10)

    # Prepare recommendations
    recommended_courses = final_rating_df[final_rating_df['course_id']
                                          .isin(top_ratings_indices)]
    top_10_recommended_courses = recommended_courses[['name', 'course_url', 'rating']].head(10)

    return top_courses_user, top_10_recommended_courses

# Streamlit UI
st.title('Sistem Rekomendasi Kursus')

# Input User ID
user_id = st.number_input('Masukkan User ID (1 - 281001):', min_value=1, max_value=num_users, step=1)

if st.button('Dapatkan Rekomendasi'):
    if user_id <= num_users:
        top_courses_user, top_10_recommended_courses = get_recommendations(user_id)
        
        st.subheader(f'Kursus Teratas yang pernah diikuti oleh {users_decoded.get(user_id)}')
        st.table(top_courses_user[['name', 'rating']].head(10))
        
        st.subheader('Rekomendasi Kursus untuk Anda')
        for idx, row in top_10_recommended_courses.iterrows():
            st.write(f"[{row['name']}]({row['course_url']}) - Rating: {row['rating']}")
    else:
        st.error("ID user tidak ditemukan.")
