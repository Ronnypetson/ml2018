import numpy as np
import pandas as pd
from sklearn import preprocessing
#from sklearn import linear_model

# Attributes of the "diamonds" table in the original order
# carat,cut,color,clarity,x,y,z,depth,table,price

# Dictionaries with the categorical values
cuts = {'Ideal':4,'Premium':3,'Good':1,'Very Good':2,'Fair':0}
colors = {'G':3,'E':5,'I':1,'J':0,'H':2,'F':4,'D':6}
clarities = {'SI1':2,'VS2':3,'SI2':1,'VS1':4,'VVS2':5,'VVS1':6,'I1':0,'IF':7}

# Converting dummy encoding to one-hot encoding for the dictionaries
#for dic in [cuts,colors,clarities]:
#	for x in dic:
#		new_x = [0.0]*len(dic)
#		new_x[dic[x]] = 1.0
#		dic[x] = new_x

# Le o arquivo e retorna os dados divididos em X e Y
def getXY(fl_path='diamonds-dataset/diamonds-train.csv'):
	# Loading the table
	diamonds_table = np.genfromtxt(fl_path,dtype=None,delimiter=',',skip_header=1)
	diamonds_table = [ list(t) for t in diamonds_table ]

	# Replacing strings by one-hot encoding
	for x in diamonds_table:
		x[1] = cuts[x[1]]
		x[2] = colors[x[2]]
		x[3] = clarities[x[3]]

	#diamonds_table = [ [x[0]]+x[1]+x[2]+x[3]+x[4:] for x in diamonds_table]

	#for x in diamonds_table:
	#	print(x)

	# Create data frame
	new_columns = ["carat","cut","color","clarity","x","y","z","depth","table","price"]
	diamonds_df = pd.DataFrame(diamonds_table,columns=new_columns)

	# Normalize columns
	diamonds_array = diamonds_df.values
	diamonds_scaled = preprocessing.MinMaxScaler().fit_transform(diamonds_array)
	diamonds_df = pd.DataFrame(diamonds_scaled,columns=new_columns)

	# ["carat","cut","color","clarity","depth","table","x","y","z"]
	# Train Sklearn
	diamonds_X = diamonds_df[["carat","cut","color","clarity","x","y","z","depth","table"]].values
	diamonds_Y = diamonds_df["price"].values

	return diamonds_X, diamonds_Y

#sk_regressor = linear_model.SGDRegressor()
#sk_regressor.fit(diamonds_X,diamonds_Y)
#print(sk_regressor.score(diamonds_X,diamonds_Y))

# Retornar np arrays X e Y

#for column in diamonds_df:
#	print(column+": %f"%diamonds_df[column].corr(diamonds_df["price"]))

