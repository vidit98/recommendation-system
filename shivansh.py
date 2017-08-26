import pandas as pd 
import numpy as np 
from collections import OrderedDict
#from sklearn.model_selection import train_test_split

test_user_id = 10917565
movies_df = pd.read_csv("movies.csv")
movies_train = pd.read_csv("training.csv")

print movies_train.loc[0]
# we convert the movie genres to a set of dummy variables 

movies_df = pd.concat([movies_df, movies_df.genres.str.get_dummies(sep='|')], axis=1)  

#print movies_df.head()
movie_categories = movies_df.columns[3:]
print movie_categories


user_preferences = OrderedDict(zip(movie_categories, []))

user_preferences['Action'] = 5  
user_preferences['Adventure'] = 5  
user_preferences['Animation'] = 1  
user_preferences["Children's"] = 1  
user_preferences["Comedy"] = 3  
user_preferences['Crime'] = 2  
user_preferences['Documentary'] = 1  
user_preferences['Drama'] = 1  
user_preferences['Fantasy'] = 5  
user_preferences['Film-Noir'] = 1  
user_preferences['Horror'] = 2  
user_preferences['Musical'] = 1  
user_preferences['Mystery'] = 3  
user_preferences['Romance'] = 1  
user_preferences['Sci-Fi'] = 5  
user_preferences['War'] = 3  
user_preferences['Thriller'] = 2  
user_preferences['Western'] =1 


def roundoff(value):
	i = int(value)
	f = value - i
	if(f >=0 and f <= .25):
		return i
	if(f >.25 and f <.75 ):
		return (i+.5)
	if( f >= .75 and f <= .99999):
		return i+1		

#in production you would use np.dot instead of writing your own dot product function.
def dot_product(vector_1, vector_2):  
    return sum([ i*j for i,j in zip(vector_1, vector_2)])


def get_movie_score(movie_features, user_preferences):  
	movies_features1 =  np.array(movie_features)
	user_preferences1 =  np.array(user_preferences)
	x1 = np.sqrt(np.sum(movies_features1**2))
	x2 = np.sqrt(np.sum(user_preferences1**2))

	return dot_product(movie_features, user_preferences)/(x1*x2)

# toy_story_features = movies_df.loc[0][movie_categories]
# toy_story_user_predicted_score = dot_product(toy_story_features, user_preferences.values())
# print toy_story_user_predicted_score

def get_movie_recommendations(user_preferences, n_recommendations):  
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
	movies_df['score'] = movies_df[movie_categories].apply(get_movie_score, args=([user_preferences]), axis=1)
	error = 0
	count =0 
	print 5*(movies_df.sort_values(by=['score'], ascending=False)['score'][:n_recommendations])
	for i in movies_train.loc[movies_train["userId"] == test_user_id]["movieId"]:
		count += 1
		val = roundoff(float(2.5*(movies_df.loc[movies_df["movieId"] == int(i)]['score']) + 2.5))
		error += abs(float((movies_train.loc[movies_train["movieId"] == int(i)].loc[movies_train["userId"] == test_user_id])['rating']) - float(val))
		print i,val
	print "errr",error/(count*5)	
	return movies_df.sort_values(by=['score'], ascending=False)['title'][:n_recommendations]

#print get_movie_recommendations(user_preferences, 10)  



def user_based_movie_recommendations(userid_given):
	#user_preferences = OrderedDict(zip(movie_categories, []))
	
	user_preferences = np.zeros(20)


	# user_preferences['Action'] = 5  
	# user_preferences['Adventure'] = 5  
	# user_preferences['Animation'] = 1  
	# user_preferences["Children's"] = 1  
	# user_preferences["Comedy"] = 3  
	# user_preferences['Crime'] = 2  
	# user_preferences['Documentary'] = 1  
	# user_preferences['Drama'] = 1  
	# user_preferences['Fantasy'] = 5  
	# user_preferences['Film-Noir'] = 1  
	# user_preferences['Horror'] = 2  
	# user_preferences['Musical'] = 1  
	# user_preferences['Mystery'] = 3  
	# user_preferences['Romance'] = 1  
	# user_preferences['Sci-Fi'] = 5  
	# user_preferences['War'] = 3  
	# user_preferences['Thriller'] = 2  
	# user_preferences['Western'] =1

	user_data = movies_train.sort_values("userId")
	user_data_temp = user_data.loc[user_data["userId"] == userid_given]
	for index, row in user_data_temp.iterrows():
		movie_id_temp = row["movieId"]
		feature1=[]
		feature1 = np.array(movies_df.loc[movies_df["movieId"] == movie_id_temp]).ravel()
		#print type(feature1)

		feature1 = feature1[3:]
		#s = np.sum(feature1)
		#print type(feature1)
		feature1 = (row["rating"] - 2.5)*feature1



		user_preferences += (feature1)


	print user_preferences
	print get_movie_recommendations(user_preferences,10)

user_based_movie_recommendations(test_user_id)			