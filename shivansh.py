import pandas as pd 
import numpy as np 
from collections import OrderedDict
#from sklearn.model_selection import train_test_split

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


#in production you would use np.dot instead of writing your own dot product function.
def dot_product(vector_1, vector_2):  
    return sum([ i*j for i,j in zip(vector_1, vector_2)])

def get_movie_score(movie_features, user_preferences):  
    return dot_product(movie_features, user_preferences)

# toy_story_features = movies_df.loc[0][movie_categories]
# toy_story_user_predicted_score = dot_product(toy_story_features, user_preferences.values())
# print toy_story_user_predicted_score

def get_movie_recommendations(user_preferences, n_recommendations):  
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
    movies_df['score'] = movies_df[movie_categories].apply(get_movie_score, 
                                                           args=([user_preferences.values()]), axis=1)
    return movies_df.sort_values(by=['score'], ascending=False)['title'][:n_recommendations]

#print get_movie_recommendations(user_preferences, 10)  



def user_based_movie_recommendations(userid_given):
	user_preferences = OrderedDict(zip(movie_categories, []))

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

	user_data = movies_train.sort_values("userid")
	user_data_temp = user_data.loc[user_data["userid == userid_given"]]
	for index, row in df.iterrows():
		movie_id_temp = row["movieid"]
