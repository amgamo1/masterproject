from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

########### Load and shape data ##################
ct2scale = np.loadtxt('ct2scale.txt')[:, 1]
ct2scale = ct2scale.reshape(1,-1)
cl2scale = np.loadtxt('cl2scale.txt')[:, 1]
cl2scale = cl2scale.reshape(1,-1)
vol2scale = np.loadtxt('./statcelldata/2scale.txt')
#vol2scale = vol2scale.reshape(1,-1)

ct09 = np.loadtxt('ct09.txt')[:, 1]
ct09 = ct09.reshape(1,-1)
cl09 = np.loadtxt('cl09.txt')[:, 1]
cl09 = cl09.reshape(1,-1)
vol09 = np.loadtxt('./statcelldata/size9.txt')
#vol09 = vol2scale.reshape(1,-1)

ct20 = np.loadtxt('ct20.txt')[:, 1]
ct20 = ct20.reshape(1,-1)
cl20 = np.loadtxt('cl20.txt')[:, 1]
cl20 = cl20.reshape(1,-1)
vol20 = np.loadtxt('./statcelldata/sphere20.txt')
#vol20 = vol20.reshape(1,-1)

ctrat = np.loadtxt('ct_ratio.txt')[:, 1]
ctrat = ctrat.reshape(1,-1)
clrat = np.loadtxt('cl_ratio.txt')[:, 1]
clrat = clrat.reshape(1,-1)
volrat = np.loadtxt('./statcelldata/ratio.txt')
#olrat = volrat.reshape(1,-1)

ctvary = np.loadtxt('ct_spherevary.txt')[:, 1]
ctvary = ctvary.reshape(1,-1)
clvary = np.loadtxt('cl_spherevary.txt')[:, 1]
clvary = clvary.reshape(1,-1)
volvary = np.loadtxt('./statcelldata/spherevary.txt')
#volvary = volvary.reshape(1,-1)

ctgg = np.loadtxt('ctgg.txt')[:, 1]
ctgg = ctgg.reshape(1,-1)
clgg = np.loadtxt('clgg.txt')[:, 1]
clgg = clgg.reshape(1,-1)
volgg = np.loadtxt('./statcelldata/n260-id260ggreg.txt')
#volgg = volgg.reshape(1,-1)

ctirreg = np.loadtxt('ctirreg.txt')[:, 1]  #Shape (261,)
ctirreg = ctirreg.reshape(-1,1)            #Data that needs to be imputed has different reshape
clirreg = np.loadtxt('clirreg.txt')[:, 1]  #Shape (261,)
clirreg = clirreg.reshape(-1,1)
volirreg = np.loadtxt('./statcelldata/irreglam.txt')  #Shape (261,)
#volirreg = volirreg.reshape(-1,1)

ctn63 = np.loadtxt('ct63lam.txt')[:, 1]      #Shape (264,)
ctn63 = ctn63.reshape(-1,1)                #Data that needs to be imputed has different reshape
cln63 = np.loadtxt('cl63lam.txt')[:, 1]      #Shape (264,)
cln63 = cln63.reshape(-1,1)                #Data that needs to be imputed has different reshape
vol63 = np.loadtxt('./statcelldata/n63lamellar.txt')     #Shape (264,)
#vol63 = vol63.reshape(-1,1)                #Data that needs to be imputed has different reshape

ctparacol = np.loadtxt('ctparacol.txt')[:, 1] #Has nan
ctparacol = ctparacol.reshape(-1,1)        #Data that needs to be imputed has different reshape
clparacol = np.loadtxt('clparacol.txt')[:, 1] #Has nan
clparacol = clparacol.reshape(-1,1)        #Data that needs to be imputed has different reshape
volparacol = np.loadtxt('./statcelldata/260columnarpara.txt') #Has nan
#volparacol = volparacol.reshape(-1,1)        #Data that needs to be imputed has different reshape

ctvoro = np.loadtxt('ctvoro.txt')[:, 1]
ctvoro = ctvoro.reshape(1,-1)
clvoro = np.loadtxt('clvoro.txt')[:, 1]
clvoro = clvoro.reshape(1,-1)
volvoro = np.loadtxt('./statcelldata/n260-id260reg.txt')
#volvoro = volvoro.reshape(1,-1)

ctZcom = np.loadtxt('ctZcolumn.txt')[:, 1] #Has nan
ctZcom = ctZcom.reshape(-1, 1)               #Data that needs to be imputed has different reshape
clZcom = np.loadtxt('clZcolumn.txt')[:, 1] #Has nan
clZcom = clZcom.reshape(-1, 1)               #Data that needs to be imputed has different reshape
volZcom = np.loadtxt('./statcelldata/n260-id260columnar.txt') #Has nan
#volZcom = volZcom.reshape(-1, 1)               #Data that needs to be imputed has different reshape

################ Dealing with NaN value ####################
ctirregnan = np.argwhere(np.isnan(ctirreg))[:,0]
clirregnan = np.argwhere(np.isnan(clirreg))[:,0]

ct63nan = np.argwhere(np.isnan(ctn63))[:,0]
cl63nan = np.argwhere(np.isnan(cln63))[:,0]

ctZcomnan = np.argwhere(np.isnan(ctZcom))[:,0]
clZcomnan = np.argwhere(np.isnan(clZcom))[:,0]

ctparacolnan = np.argwhere(np.isnan(ctparacol))[:,0]
clparacolnan = np.argwhere(np.isnan(clparacol))[:,0]

# Defining imputer model
imp_tree = IterativeImputer(random_state=0, missing_values= np.nan,
                            estimator=ExtraTreesRegressor(max_features="sqrt", random_state=0))
ctirreg = imp_tree.fit_transform(ctirreg)
clirreg = imp_tree.fit_transform(clirreg)

ctn63 = imp_tree.fit_transform(ctn63)
cln63 = imp_tree.fit_transform(cln63)

ctZcom = imp_tree.fit_transform(ctZcom)
clZcom = imp_tree.fit_transform(clZcom)

ctparacol = imp_tree.fit_transform(ctparacol)
clparacol = imp_tree.fit_transform(clparacol)

# Dropping excess samples
ctn63drop = [ct63nan[1],ct63nan[2], ct63nan[3], ct63nan[4]]
cln63drop = [cl63nan[1],cl63nan[2], cl63nan[3], cl63nan[4]]
voln63drop = [cl63nan[1],cl63nan[2], cl63nan[3], cl63nan[4]]

ctirregdrop = [ctirregnan[1]]
clirregdrop = [clirregnan[1]]
volirregdrop = [clirregnan[1]]

ctirreg = np.delete(ctirreg, ctirregdrop)
clirreg = np.delete(clirreg, clirregdrop)
volirreg = np.delete(volirreg, volirregdrop, 0)



ctn63 = np.delete(ctn63, ctn63drop)
cln63 = np.delete(cln63, cln63drop)
vol63 = np.delete(vol63, voln63drop, 0)


#Reshape the imputed data back to the correct shape for grain = sample
ctirreg = ctirreg.reshape(1,-1)
clirreg = clirreg.reshape(1,-1)
#volirreg = volirreg.reshape(-1,1)

ctn63 = ctn63.reshape(1,-1)
cln63 = cln63.reshape(1,-1)
#vol63 = vol63.reshape(-1,1)


ctZcom = ctZcom.reshape(1,-1)
clZcom = clZcom.reshape(1,-1)
#volZcom = volZcom.reshape(1,-1)

ctparacol = ctparacol.reshape(1,-1) #test
clparacol = clparacol.reshape(1,-1) #test
volparacol = volparacol.reshape(1,-1) #test


allct = np.concatenate([ct2scale.T, ct20.T, ctirreg.T, ctZcom.T, ct09.T, ctn63.T, ctgg.T, ctvoro.T, ctrat.T,ctvary.T])
allcl = np.concatenate([cl2scale.T, cl20.T, clirreg.T, clZcom.T, cl09.T, cln63.T, clgg.T, clvoro.T, clrat.T,clvary.T])
allmicro = np.concatenate([vol2scale, vol20, volirreg, volZcom, vol09, vol63, volgg, volvoro, volrat,volvary],axis = 0)


#Scaling the data
scaling_input = MinMaxScaler()
scaling_output = MinMaxScaler()

#Test data
scaletest_input = MinMaxScaler()
scaletest_output = MinMaxScaler()
ctpara = np.concatenate([ctparacol.T], axis = 1)
clpara = np.concatenate([clparacol.T], axis = 1)
Y_unseen = pd.DataFrame(data = clpara)
X_unseen = pd.DataFrame(data = volparacol)

test_scaled = scaletest_input.fit_transform(X_unseen)
Y_unseen_scaled = scaletest_output.fit_transform(Y_unseen)

#Final training frame
Y_cl = np.concatenate([allcl], axis = 1)
Y_output = pd.DataFrame(data = Y_cl )
Y_scaled = scaling_output.fit_transform(Y_output)
#print(Y_scaled)
X = pd.DataFrame(data = allmicro)
X_scaled = scaling_input.fit_transform(X)
#X_scaled = X
#print(X_scaled)

#Build the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform', activation='sigmoid'))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(#loss = tf.keras.losses.MeanAbsolutePercentageError(),
                  loss = 'mse',
                  optimizer = 'adam' ,
                  metrics = ['mae'])
                  #metrics  = [tf.keras.metrics.MeanAbsolutePercentageError()])
    return model

def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		#print('>%.3f' % mae)
		results.append(mae)
	return results

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=20)
model = get_model(X.shape[1],Y_scaled.shape[1])
history = model.fit(X_train, y_train,
                    validation_data= (X_test, y_test),
                    epochs = 50)

#results = evaluate_model(X_test, y_test)
#print('MAE: %.3f (%.3f)' % (mean(results), std(results)))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['mae']
val_accuracy = history_dict['val_mae']

epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#
# Plot the model accuracy (MAE) vs Epochs
#
ax[0].plot(epochs, accuracy, 'bo', label='Training mape')
ax[0].plot(epochs, val_accuracy, 'b', label='Validation mape')
ax[0].set_title('Training & Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuracy', fontsize=16)
ax[0].legend()
#
# Plot the loss vs Epochs
#
ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()

plt.show()
