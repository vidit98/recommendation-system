import pandas as pd 
import numpy as np 
import csv

movies_df = pd.read_csv("movies.csv")
movies_train = pd.read_csv("training.csv")

movies_df = pd.concat([movies_df, movies_df.genres.str.get_dummies(sep='|')], axis=1)  
movie_categories = movies_df.columns[4:]
test_movies_id = []
test_movies=[]
popular= pd.read_csv("popular.csv")
del popular["a"]
#popular = popular.head(n=350)
#print popular.shape
"""def add_count():
	for i in movies_df["movieId"]:
		print i, type(i)
		if(i in list(movies_train["movieId"].values)):
			movies_df.loc["movieId" == int(i)]["count"] = len(movies_train.loc[movies_train["movieId"] == i])

	
	movies_df1 = movies_df.copy()
	movies_df1["count"] = 0

	for index, row in movies_df1.iterrows():
		movie_id_temp = row["movieId"]
		movies_df1.ix[index, "count" ] = len(movies_train.loc[movies_train["movieId"] == movie_id_temp])

	res = movies_df1.sort_values('count', ascending=False)	
	res.to_csv("popular.csv")

#print movies_df["count"]
"""
def find_year(array):

	idx = array[1].find(")" , len(array[1]) - 6)
	try:
		y = array[1][idx-4:idx]
		return int(y)
	except:
		return 0	


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



def get_movie_recommendations(user_preferences, m_seen , avg,var):  
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
	#movies_df1 = movies_df.copy()
	#test_movies1 = pd.DataFrame(data = test_movies, columns = movies_df.columns)
	#test_categories = test_movies1.columns[4:]
	start = int(round(avg) - np.ceil(var))
	end = int(round(avg) + np.ceil(var))
	#test_movies1['score'] = test_movies1[test_categories].apply(get_movie_score, args=([user_preferences]), axis=1)
	
	popular1 = popular.copy()
	#popular_categories = popular1.columns[4:]
	#popular1['score'] = popular1[popular_categories].apply(get_movie_score, args=([user_preferences]), axis=1)

	popular2 = (popular1.head(n =1000)).copy()
	popular_categories2 = popular2.columns[4:]
	popular2['score'] = popular2[popular_categories2].apply(get_movie_score, args=([user_preferences]), axis=1)
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

	#result = test_movies1.sort_values('score')

	result = popular2.sort_values('score', ascending=False)
	#print result

	c = 1

	for index, row in result.iterrows():
		#print row
		y = find_year(np.array(row))
		if((row["movieId"] in m_seen) or (y not in range(start,end+1))):
			 result.drop(index, inplace=True)
		else:
			c += 1

		if c == 6:
			break					 

	if c == 6:
		#print "if"
		result = result.head()
		return np.array(result[['movieId', 'title','score']])	
	else:

		no = c
		popular1 = popular1.tail(len(popular1) - 1000)

		for index, row in popular1.iterrows():
			y = find_year(np.array(row))
			if((row["movieId"] in m_seen) or (y not in range(start,end+1))):
				 popular1.drop(index, inplace=True)
			else:
				c += 1

			if c == 6:
				#print "else"
				break					 

		add = popular1.head(n=6-no).copy()
		add_categories = add.columns[4:]
		add['score'] = add[add_categories].apply(get_movie_score, args=([user_preferences]), axis=1)
		result = result.append(add)
	#print add
	return np.array(result[['movieId', 'title','score']])			
	#return np.array(result[['movieId', 'title','score']])
	#return result


def user_based_movie_recommendations(userid_given, missing_movies):

	user_preferences = np.zeros(19)

	user_data = movies_train.sort_values("userId")
	user_data_temp = user_data.loc[user_data["userId"] == userid_given]
	movies_seen = np.array(user_data_temp["movieId"])

	s = user_data_temp["rating"].sum()

	s = s/len(user_data_temp["rating"])
	year = []
	for index, row in user_data_temp.iterrows():
		movie_id_temp = row["movieId"]

		#if(movies_df.loc["movieId" == movie_id_temp]["genres"]):
		if(movie_id_temp not in missing_movies):
			feature1=[]
			
			feature1 = np.array(movies_df.loc[movies_df["movieId"] == movie_id_temp]).ravel()
			y = find_year(feature1)
			year.append(y)
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
	avg = np.mean(year)
	var= np.var(year)
	return get_movie_recommendations(user_preferences, movies_seen , avg , np.sqrt(var)), s



	
missing_movies = remove_no_genre()
#extract_2015()
#add_count()

ofile  = open('result1.csv', "w")
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

			
			#t = movies_train.loc[movies_train["userId"] == user_id[0]]
			#movies_seen = np.array(t["movieId"])

			movies, s =  user_based_movie_recommendations(int(user_id[0]), missing_movies)
			"""movies["val"] = 0

			for index, row in movies.iterrows():

				score = row["score"]
				row = np.array(row)[4:-2]
				val = 0
				for index2, row2 in popular.iterrows():
					row2 = np.array(row2)[4:-1]
					val += get_movie_score(row, row2)

				movies.ix[index, "val"] = val*0 + score

			res = np.array(movies.sort_values("val").tail())"""

		except ZeroDivisionError:
			for i in range(5):
				rating = 2.5
				ans = [user_id[0] , result[i,0] , rating]
				writer.writerow(ans)
			continue


		for i in range(len(movies)):
			rating = roundoff(abs(float((5.0-s)*movies[i,2] + s)))
			ans =[ user_id[0] , movies[i, 0], rating ]
			writer.writerow(ans)
			ofile.flush()

f.close()
ofile.close()			

#movies, s = user_based_movie_recommendations(int(user_id[0]), missing_movies)

#movies_watch = np.zeros(len(movies_df))

"""movies, s =  user_based_movie_recommendations(10815242, missing_movies)
movies["val"] = 0

for index, row in movies.iterrows():

	score = row["score"]
	row = np.array(row)[4:-2]
	val = 0
	for index2, row2 in popular.iterrows():
		row2 = np.array(row2)[4:-1]
		val += dot_product(row, row2)*score

	movies.ix[index, "val"] = val 

res = movies.sort_values("val").tail()


#print res[:,["movieId", "title","score"]]
"""



