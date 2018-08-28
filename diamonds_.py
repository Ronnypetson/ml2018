import numpy as np
import pandas as pd

# Attributes of the "diamonds" table in the original order
# "","carat","cut","color","clarity","depth","table","price","x","y","z"
#       0     1-5    6-12   13-20      21      22      23    24  25  26

# Dictionaries with the categorical values
cuts = {'"Ideal"':4,'"Premium"':3,'"Good"':1,'"Very Good"':2,'"Fair"':0}
colors = {'"G"':3,'"E"':5,'"I"':1,'"J"':0,'"H"':2,'"F"':4,'"D"':6}
clarities = {'"SI1"':2,'"VS2"':3,'"SI2"':1,'"VS1"':4,'"VVS2"':5,'"VVS1"':6,'"I1"':0,'"IF"':7}

# Converting dummy encoding to one-hot encoding for the dictionaries
#for dic in [cuts,colors,clarities]:
#	for x in dic:
#		new_x = [0.0]*len(dic)
#		new_x[dic[x]] = 1.0
#		dic[x] = new_x

# Loading the table
diamonds_table = np.genfromtxt('diamonds.csv',dtype=None,delimiter=',',skip_header=1,usecols=range(1,11))
diamonds_table = [ list(t) for t in diamonds_table ]

# Replacing strings by one-hot encoding
for x in diamonds_table:
	x[1] = cuts[x[1]]
	x[2] = colors[x[2]]
	x[3] = clarities[x[3]]

#diamonds_table = [ [x[0]]+x[1]+x[2]+x[3]+x[4:] for x in diamonds_table]

for x in diamonds_table:
	print(x)

new_columns = ["carat","cut","color","clarity","depth","table","price","x","y","z"]
diamonds_df = pd.DataFrame(diamonds_table,columns=new_columns)

for column in diamonds_df:
	print(column+": %f"%diamonds_df[column].corr(diamonds_df["price"]))

