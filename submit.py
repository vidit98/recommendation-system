import pandas as pd 
import numpy as np 
import csv

movies_df = pd.read_csv("movies.csv")
movies_train = pd.read_csv("training.csv")

movies_df = pd.concat([movies_df, movies_df.genres.str.get_dummies(sep='|')], axis=1)  
movie_categories = movies_df.columns[4:]
test_movies_id = []
test_movies=[]

def remove_no_genre():

	missing_movies=[]

	for i in np.where(movies_df["(no genres listed)"] == 1):

		missing_movies.append(movies_df.iloc[i,0].values)
		movies_df.drop(movies_df.index[i], inplace=True)

	movie_categories = movies_df.columns[4:]	

	return np.array(missing_movies).ravel()
	


def extract_2015():

	for i in movies_df["title"]:
		if i.find("(2015)") != -1:
			test_movies.append(np.array(movies_df[movies_df["title"]== i].values).ravel())
			#print i
			#print movies_df[movies_df["title"] == i]["movieId"]
			#print movies_df[movies_df["title"]== "Jupiter Ascending (2015)"]["movieId"][0]
			test_movies_id.append(movies_df[movies_df["title"]== i].values[0][0])


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
	
	movie_features1 =  np.array(movie_features)
	
	user_preferences1 =  np.array(user_preferences )
	x1 = np.sqrt(np.sum(movie_features1**2))
	x2 = np.sqrt(np.sum(user_preferences1**2))
	
	return dot_product(movie_features, user_preferences)/(x1*x2)



def get_movie_recommendations(user_preferences, n_recommendations,s):  
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
	#movies_df1 = movies_df.copy()
	test_movies1 = pd.DataFrame(data = test_movies, columns = movies_df.columns)
	test_categories = test_movies1.columns[4:]

	test_movies1['score'] = test_movies1[test_categories].apply(get_movie_score, args=([user_preferences]), axis=1)
	#error = 0
	#count =0 
	#print 5*(movies_df.sort_values(by=['score'], ascending=False)['score'][:n_recommendations])
	#for i in test_movies1["movieId"]:
		#if i in list(user_test["movieId"]):
		#count += 1
		#val=(roundoff(abs(float((5-s)*(test_movies1.loc[test_movies1["movieId"] == int(i)]['score']) + s))))
		#print float((5-s)*(movies_df1.loc[movies_df1["movieId"] == int(i)]['score']) + s)
		#error += abs(float((movies_train.loc[movies_train["movieId"] == int(i)].loc[movies_train["userId"] == test_user_id])['rating']) - float(val))
	

	#print (test_user_id,error/(count*5))
	#return (test_user_id,error/(count*5))	

	#for index, row in test_movies1.iterrows():
		#print roundoff(abs(float((5.0-s)*row["score"] + s)))
		#test_movies1.loc[test_movies1["movieId"] == int(row["movieId"])].loc[1, 21]= roundoff(abs(float((5.0-s)*row["score"] + s)))

	result = test_movies1.sort_values('score')
	return np.array(result.tail()[['movieId', 'title','score']])


def user_based_movie_recommendations(userid_given, missing_movies):

	user_preferences = np.zeros(19)

	user_data = movies_train.sort_values("userId")
	user_data_temp = user_data.loc[user_data["userId"] == userid_given]
	s = user_data_temp["rating"].sum()

	s = s/len(user_data_temp["rating"])

	for index, row in user_data_temp.iterrows():
		movie_id_temp = row["movieId"]
		#if(movies_df.loc["movieId" == movie_id_temp]["genres"]):
		if(movie_id_temp not in missing_movies):
			feature1=[]
			
			feature1 = np.array(movies_df.loc[movies_df["movieId"] == movie_id_temp]).ravel()
		
			feature1 = feature1[4:]
		
			#s = np.sum(feature1)
			#print type(feature1)
			feature1 = (row["rating"] - s)*feature1



			user_preferences += (feature1)

	if(np.sum(user_preferences**2) == 0):
		for index, row in user_data_temp.iterrows():
			
			if(movie_id_temp not in missing_movies):

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
	return get_movie_recommendations(user_preferences, 5, s), s



	
missing_movies = remove_no_genre()
extract_2015()

ofile  = open('result.csv', "w")
writer = csv.writer(ofile)

movies_df1 = movies_df.copy()
movies_df1["no of genres"]= np.sum(movies_df1, axis =1)
result = np.array(movies_df1.sort_values('no of genres').tail()[['movieId', 'title']])


with open("test.csv") as f:
	reader = csv.reader(f)
	#reader = reader[0:5]
	c = 0

	for row in reader:
		c+=1
		if c == 1:
			continue
		user_id = row
		print user_id[0],c

		try:
			movies, s =  user_based_movie_recommendations(int(user_id[0]), missing_movies)
		except ZeroDivisionError:
			for i in range(5):
				rating = 2.5
				ans = [user_id[0] , result[i,0] , rating]
				writer.writerow(ans)
			continue


		for i in range(len(movies)):
			rating = roundoff(abs(float((5.0-s)*movies[i, 2] + s)))
			ans =[ user_id[0] , movies[i, 0], rating ]
			writer.writerow(ans)

f.close()
ofile.close()			
 
