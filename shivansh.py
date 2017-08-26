import pandas as pd 
import numpy as np 
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import correlation


test_user_id = 10917565


movies_df = pd.read_csv("movies.csv")
movies_train = pd.read_csv("training.csv")


#print movies_train.loc[0]
# we convert the movie genres to a set of dummy variables 

movies_df = pd.concat([movies_df, movies_df.genres.str.get_dummies(sep='|')], axis=1)  

#print movies_df.head()
movie_categories = movies_df.columns[4:]

distinct_users = movies_train["userId"].unique()
test_user_id_array =  distinct_users[:300]

cor = correlation.weights()

user_preferences = OrderedDict(zip(movie_categories, []))



def roundoff(value):
	i = int(value)
	f = value - i
	if(f >=0 and f <= .25):
		return i
	if(f >.25 and f <.75 ):
		return (i+.5)
	if( f >= .75):
		return i+1		


#in production you would use np.dot instead of writing your own dot product function.
def dot_product(vector_1, vector_2):  
    return sum([ i*j for i,j in zip(vector_1, vector_2)])


def get_movie_score(movie_features, user_preferences): 
	#movie_features = np.array(movie_features)
	
	movie_features1 =  np.array(movie_features)
	
	user_preferences1 =  np.array(user_preferences )
	x1 = np.sqrt(np.sum(movie_features1**2))
	x2 = np.sqrt(np.sum(user_preferences1**2))
	
	return dot_product(movie_features, user_preferences)/(x1*x2)

# toy_story_features = movies_df.loc[0][movie_categories]
# toy_story_user_predicted_score = dot_product(toy_story_features, user_preferences.values())
# print toy_story_user_predicted_score

def get_movie_recommendations(user_preferences, n_recommendations,s):  
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
	movies_df1 = movies_df.copy()
	
	movies_df1['score'] = movies_df1[movie_categories].apply(get_movie_score, args=([user_preferences]), axis=1)
	error = 0
	count =0 
	#print 5*(movies_df.sort_values(by=['score'], ascending=False)['score'][:n_recommendations])
	for i in movies_train.loc[movies_train["userId"] == test_user_id]["movieId"]:
		#if i in list(user_test["movieId"]):
		count += 1
		val = roundoff(abs(float((5-s)*(movies_df1.loc[movies_df1["movieId"] == int(i)]['score']) + s)))
		#print float((5-s)*(movies_df1.loc[movies_df1["movieId"] == int(i)]['score']) + s)
		error += abs(float((movies_train.loc[movies_train["movieId"] == int(i)].loc[movies_train["userId"] == test_user_id])['rating']) - float(val))
		
	print (test_user_id,error/(count*5))
	return (test_user_id,error/(count*5))	
	#return movies_df.sort_values(by=['score'], ascending=False)['title'][:n_recommendations]


#print get_movie_recommendations(user_preferences, 10)  



def user_based_movie_recommendations(userid_given):

	#user_preferences = OrderedDict(zip(movie_categories, []))
	
	user_preferences = np.zeros(19)




	user_data = movies_train.sort_values("userId")
	user_data_temp = user_data.loc[user_data["userId"] == userid_given]
	user_train , user_test = train_test_split(user_data_temp , test_size = .2)
	s = user_data_temp["rating"].sum()
	s = s/len(user_data_temp["rating"])


	print s , "ssss"

	for index, row in user_data_temp.iterrows():
		movie_id_temp = row["movieId"]
		#if(movies_df.loc["movieId" == movie_id_temp]["genres"]):
		feature1=[]
		
		feature1 = np.array(movies_df.loc[movies_df["movieId"] == movie_id_temp]).ravel()
		
		feature1 = feature1[4:]
		
		#s = np.sum(feature1)
		#print type(feature1)
		feature1 = (row["rating"] - s)*feature1



		user_preferences += (feature1)

	if(np.sum(user_preferences**2) == 0):
		for index, row in user_data_temp.iterrows():
			movie_id_temp = row["movieId"]
			#if(movies_df.loc["movieId" == movie_id_temp]["genres"]):
			feature1=[]
			
			feature1 = np.array(movies_df.loc[movies_df["movieId"] == movie_id_temp]).ravel()
			
			feature1 = feature1[4:]
			
			#s = np.sum(feature1)
			#print type(feature1)
			feature1 = (row["rating"] - 2.49)*feature1



			user_preferences += (feature1)

			s = 2.49

	#print user_preferences
	#print user_test
	return get_movie_recommendations(user_preferences,10 , s)
error_array = []
c = 0
for test_user_id in test_user_id_array[200:300]:
	c+=1

	print test_user_id,c
	error_array.append(user_based_movie_recommendations(test_user_id)[1])
#user_based_movie_recommendations(test_user_id)	
print sum(error_array)