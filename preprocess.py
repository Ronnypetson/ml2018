import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# Attributes of the "diamonds" table in the original order
# carat,cut,color,clarity,x,y,z,depth,table,price

# Dictionaries with the categorical values
cuts = {'Ideal':4,'Premium':3,'Good':1,'Very Good':2,'Fair':0}
colors = {'G':3,'E':5,'I':1,'J':0,'H':2,'F':4,'D':6}
clarities = {'SI1':2,'VS2':3,'SI2':1,'VS1':4,'VVS2':5,'VVS1':6,'I1':0,'IF':7}

# Converting dummy encoding to one-hot encoding for the dictionaries
for dic in [cuts,colors,clarities]:
	for x in dic:
		new_x = [0.0]*len(dic)
		new_x[dic[x]] = 1.0
		dic[x] = new_x

# Le o arquivo e retorna os dados divididos em X e Y
def getXY(fl_path='diamonds-dataset/diamonds-train.csv',train_norm=False,col_means=None,col_var=None):
	# Loading the table
	diamonds_table = np.genfromtxt(fl_path,dtype=None,delimiter=',',skip_header=1,encoding='ascii')
	diamonds_table = [ list(t) for t in diamonds_table ]

	# Replacing strings by one-hot encoding
	for x in diamonds_table:
		x[1] = cuts[x[1]]
		x[2] = colors[x[2]]
		x[3] = clarities[x[3]]

	diamonds_table = [ [x[0]]+x[1]+x[2]+x[3]+x[4:] for x in diamonds_table] # One-hot mode

	# Create data frame
	new_columns = ["carat","cut0","cut1","cut2","cut3","cut4",\
											 "color0","color1","color2","color3","color4","color5","color6",\
											 "clarity0","clarity1","clarity2","clarity3","clarity4","clarity5","clarity6","clarity7",\
											 "x","y","z","depth","table","price"] # One-hot mode
	#new_columns = ["carat","cut","color","clarity","x","y","z","depth","table","price"]
	diamonds_df = pd.DataFrame(diamonds_table,columns=new_columns)
	#diamonds_df["carat_2"] = diamonds_df["carat"]**1.5
	#diamonds_df["carat_3"] = diamonds_df["carat"]**3
	#diamonds_df["carat_exp"] = np.exp(0.5*diamonds_df["carat"])
	#sns.heatmap(data=diamonds_df[new_columns].corr(),annot=True,cbar=True)
	#plt.show()
	new_columns = [col for col in new_columns if col not in ["depth","table","price"]]
	
	# Normalize columns
	diamonds_array = diamonds_df.values
	np.random.shuffle(diamonds_array)
	if train_norm:
		std_scaler = preprocessing.StandardScaler()
		diamonds_scaled = std_scaler.fit_transform(diamonds_array)
		diamonds_df = pd.DataFrame(diamonds_scaled,columns=list(diamonds_df))
		diamonds_X = diamonds_df[new_columns].values
		diamonds_Y = diamonds_df["price"].values
		diamonds_X = np.insert(diamonds_X,0,1,1)
		return diamonds_X, diamonds_Y, std_scaler.mean_, std_scaler.var_
	else:
		diamonds_scaled = np.array([(x-col_means)/np.sqrt(col_var) for x in diamonds_array])
		diamonds_df = pd.DataFrame(diamonds_scaled,columns=list(diamonds_df))
		diamonds_X = diamonds_df[new_columns].values
		diamonds_Y = diamonds_df["price"].values
		diamonds_X = np.insert(diamonds_X,0,1,1)
		return diamonds_X, diamonds_Y

#for column in diamonds_df:
#	print(column+": %f"%diamonds_df[column].corr(diamonds_df["price"]))
## Create artificial data
#train_X = np.random.uniform(-1.0,1.0,size=(100,10))
#train_X = np.insert(train_X,0,1,1)
#theta = np.random.normal(1.0,5.0,size=train_X.shape[1])
#train_Y = np.dot(train_X,theta) + np.random.normal(0.0,0.1,size=100)
#plt.plot(train_X,train_Y,'ro')
#plt.show()
#diamonds_X = diamonds_df.loc[:,diamonds_df.columns != "price"].values # One-hot mode

