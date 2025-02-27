"""
SXOLIA GIA TON UPOLOGISMO TOU MAPE
gia ton upologismo tou percentage error, an h actual timi einai miden, tote afou den mporoume na diairesoume me to miden,
diairoume me ti mesi timh twn actual timwn se ekeinh th sthlh,
ta borousame na thewrisoume to posostiaio sfalma 0, h kapoia allh timh
"""
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras import layers, optimizers, losses, metrics

model = load_model('./H5/WindDenseNN.h5', compile=False)
print('loaded model.')
# summarize the structure of the model
summary = model.summary()
# Get the weights of the first layer of the model, theloume mono tou 1ou layer ta varh, afou auto tha anakataskeuasoume
weights = model.layers[0].get_weights()
#print("debug, weights of first layer")
#print(weights)

"""
test_data = pd.read_csv('./CSVs/nn_representations.csv', header=None,index_col=0)
print('test data head indexcol = 0')
print(test_data.head())
test_data = pd.read_csv('./CSVs/nn_representations.csv',header=None,index_col=False)
print('test data head false')
print(test_data.head())
test_data = pd.read_csv('./CSVs/nn_representations.csv',header=None,index_col=None)
print('test data head none')
print(test_data.head())
"""


#svinoume tin prwti stili pou exei to data
test_data = pd.read_csv('./CSVs/nn_representations.csv', header=None)
print('test data head prin to drop')
print(test_data.head())

#dropping first column
# If you know the name of the column skip this
first_column = test_data.columns[0]
# Delete first
test_data = test_data.drop([first_column], axis=1)
print('test data head')

print(test_data.head())

test_data.to_csv("temp.csv", index=False, sep=' ', encoding='utf-8')

with open("temp.csv",'r') as f:
    with open("nn_representations_cleaned.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)


result = model.predict(test_data, batch_size=32)
print("predicted, now printing shape")
print('prediction result shape : ',result.shape)
#print(result)

actual_results = pd.read_csv('./CSVs/actual.csv')
actual_results = actual_results.drop(actual_results.columns[[0]], axis=1)

#print(actual_results)



# ERWTHMA A
err = np.zeros((actual_results.shape[0],7))
percent_err = np.zeros((actual_results.shape[0],7))
sq_err = np.zeros((actual_results.shape[0],7))

for i in range(err.shape[0]):
	for j in range(7):
		err[i][j] = abs(actual_results.iloc[i,j] - result[i][j] )
		if actual_results.iloc[i,j] == 0:
			percent_err[i][j] =  result[i][j] / np.mean(actual_results.iloc[:,j])
		else:
			percent_err[i][j] = err[i][j] /  (actual_results.iloc[i,j] )
		sq_err[i][j] = (err[i][j]) ** 2

mae = np.mean(err)
mape = np.mean(percent_err)
mse = np.mean(sq_err)

print('\nMAE =',mae, ' MAPE =',mape,' MSE = ', mse, '\n' )
print('---')

# TELOS ERWTHMATOS A


# theloume na ftiaksoume to N2 xwris to 2o layer tou.
print('\n creating new model\n')
new_model = keras.Sequential()
for layer in model.layers[:-1]:
    new_model.add(layer)

#pws akrivws kserw ta losses, metrics, optimizers pou eixe xrisimopoiisei autos gia to compile tou montelou tou?
new_model.compile(optimizer=optimizers.RMSprop(0.01),loss=losses.CategoricalCrossentropy(),metrics=[metrics.CategoricalAccuracy()])


print('\nmodel summary')
print(model.summary)
print('\nnew model summary')
print(new_model.summary)
print('\ndebug, model layers are:')
print(model.layers)
print('debug, new model layers are:')
print(new_model.layers)


result = new_model.predict(test_data, batch_size=32)
print('predicted with new model')

test_data = pd.read_csv('./CSVs/nn_representations.csv')
#svinoume tin prwti stili pou exei to data
test_data = test_data.drop(test_data.columns[[0]], axis=1)

result = new_model.predict(test_data, batch_size=32)
print("predicted with new model, now printing shape")
print('prediction result shape : ',result.shape)

print('predicted with new model')
print(type(result))
np.savetxt('result.out', result)
