import streamlit as st
import streamlit.components.v1 as components 
import pickle # to save and load models
import numpy as np
import pandas as pd
import difflib # helpers for computing deltas
import random
# imports form surprise
from surprise import accuracy, Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from collections import defaultdict

siteHeader = st.container()
model_prediction = st.container()
dataset = st.container()
new_features = st.container()

@st.cache_data
def get_data1():
    recsys_df = pd.read_csv('./recsys_df_update.csv')
    return recsys_df

recsys_df = get_data1()

@st.cache_data
def get_data2():
    instagram_data = pd.read_csv("./posts_0.csv", engine='python')
    return instagram_data

instagram_data = get_data2()

@st.cache_data
def get_algo():
    algo = pickle.load(open('./model.pkl', 'rb'))
    return algo

algo = get_algo()

# Functions to get location recommendations
def get_location_id(name, metadata):
    
    """
    Gets the location ID for a location name based on the closest match in the metadata dataframe
    """
    
    existing_names = list(metadata['name'].values)
    closest_names = difflib.get_close_matches(name, existing_names)
    new_location_id = metadata[metadata['name'] == closest_names[0]]['new_location_id'].values[0]
    return new_location_id

def get_location_info(new_location_id, metadata):
    
    """
    Returns some basic information about a location given the location id and the metadata dataframe
    """
    
    location_info = metadata[metadata['new_location_id'] == new_location_id][['new_location_id', 'name', 
                                                    'city', 'cd']]
    return location_info.to_dict(orient='records')

def predict_rating(profile_id, name, model, metadata):
    
    """
    Predicts the review (on a scale of 1-3) that a user would assign to a specific location 
    """
    
    new_location_id = get_location_id(name, metadata)
    rating_prediction = algo.predict(uid=profile_id, iid=new_location_id)
    return rating_prediction.est

def generate_recommendation(profile_id, model, metadata, thresh=2.5):
    
    """
    Generates a location recommendation for a user based on a rating threshold. Only
    books with a predicted rating at or above the threshold will be recommended
    """
    
    if profile_id in metadata['profile_id'].values:
        names = list(metadata['name'].values)
        random.shuffle(names)
        for name in names:
            rating = predict_rating(profile_id, name, model, metadata)
            if rating >= thresh:
                new_location_id = get_location_id(name, metadata)
                return get_location_info(new_location_id, metadata)[0]
    else:
        # counter cold start problem by recommending top 20 rated locations
        st.write("Looks like you're not a member yet. Why not join now for better recommendation?")
        names = recsys_df.groupby(['name']).count().sort_values(by='profile_id', ascending=False).head(20).index.values
        random.shuffle(names)
        for name in names:
            new_location_id = get_location_id(name, metadata)
            return (get_location_info(new_location_id, metadata)[0])

with siteHeader:
    st.title("Let's explore the hidden gems of London!")
    st.text("In this project, I've looked into an instagram dataset,")
    st.text("filtered the usual touristy spots out to give you the hidden gems of London.")
    
with model_prediction:
    st.header("Let's Explore!")
    st.header("Find your Instagram user id using the website below")
    components.iframe("https://www.instafollowers.co/find-instagram-user-id")
    st.text("Input your profile id and you're good to go!")
    
    # Get user inputs
    profile_id = st.number_input("profile id:", min_value=0, help="this is your membership id") 
    
    # Add a submit button
    if st.button("Submit"):
        st.write(generate_recommendation(profile_id, algo, recsys_df))
    
with dataset:
    st.header("Dataset: Instagram posts")
    st.text("I found this dataset from Kaggle and I've decided to work with it")
    st.text("because I find that social media has a great influence on")
    st.text("where we decide to go for our holidays.")
    st.write(instagram_data.head())
    
with new_features:
    st.header("New features I came up with")
    st.markdown("* **user rating:** based on sentiment analysis on the caption of the Instagram posts, I've feature engineered the user rating on the location tagged")