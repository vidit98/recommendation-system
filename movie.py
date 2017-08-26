import movie_feature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def get_movie_feature(user):
	"""
	Returns the feature vector of movies that user has watched. User is passed as argument
	Some movies that user watched but genre is not available is stored in missing_movies
	Does not seperate (no genre listed) movie can be seperated by feature having index 14 == 1
	"""
	input_file = "movies.csv"
	genres_data = pd.read_csv(input_file)
	feature = []
	for i in genres_data["genres"]:
		feature = feature + i.split("|")
	dist_feature = set(feature)	
	dist_feature = list(dist_feature)
	dist_feature.pop(14)
	feature_vector = [] #20 x no of movies user rated
	missing_movies = set(user["movieId"]).difference(genres_data["movieId"])
	
	for i in genres_data["movieId"]:
		if i in list(user["movieId"]):
			feature1=[]
			feature1 = str(genres_data.loc[genres_data["movieId"] == i]["genres"]).split("|")

			feature_vector.append([1 if j in feature1 else 0 for j in dist_feature ])
		
			
	return feature_vector,missing_movies		
	

input_file1 = "training.csv"
user_data = pd.read_csv(input_file1)

user = user_data.loc[user_data["userId"] == 14909330] #currently training for a particular user

print len(user)

user_train , user_test = train_test_split(user , test_size = .2)
user_train = user_train.sort_values("movieId")
user_test = user_test.sort_values("movieId")

train_feature , missing_movies= get_movie_feature(user_train)

train_feature = np.dot(train_feature , movie_feature.weights())

ratings = []


for i in user_train["movieId"]:
	if i not in missing_movies:
		ratings.append(user_train.loc[user_train["movieId"] == i]["rating"])

test_feature = get_movie_feature(user_test)[0]

linear = LinearRegression()

linear.fit(train_feature ,ratings)
print user_train
print user_test

print linear.predict(test_feature)







	