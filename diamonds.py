import numpy as np

# Attributes of the "diamonds" table in the original order
# "","carat","cut","color","clarity","depth","table","price","x","y","z"

# Dictionaries with the categorical values
cuts = {'"Ideal"':0,'"Premium"':1,'"Good"':2,'"Very Good"':3,'"Fair"':4}
colors = {'"G"':0,'"E"':1,'"I"':2,'"J"':3,'"H"':4,'"F"':5,'"D"':6}
clarities = {'"SI1"':0,'"VS2"':1,'"SI2"':2,'"VS1"':3,'"VVS2"':4,'"VVS1"':5,'"I1"':6,'"IF"':7}

# Converting dummy encoding to one-hot encoding for the dictionaries
for dic in [cuts,colors,clarities]:
	for x in dic:
		new_x = [0.0]*len(dic)
		new_x[dic[x]] = 1.0
		dic[x] = new_x

# Loading the table
diamonds_table = np.genfromtxt('diamonds.csv',dtype=None,delimiter=',',skip_header=1,usecols=range(1,11))
diamonds_table = [ list(t) for t in diamonds_table ]

# Replacing strings by one-hot encoding
for x in diamonds_table:
	x[1] = cuts[x[1]]
	x[2] = colors[x[2]]
	x[3] = clarities[x[3]]

diamonds_table = [ [x[0]]+x[1]+x[2]+x[3]+x[4:-1] for x in diamonds_table]

for x in diamonds_table:
	print(x)

