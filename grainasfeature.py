############### Import Packages ###################################
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
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


########### Load and shape data ##################
# All datas are reshaped (1,-1) to be (no of grains, 1) with 260 rows and 1 column.
# In data shape terms, it means 260 samples and 1 feature. Each feature is average c per grain
# Each sample is a grain in the microstructure.
# A grain cannot be a feature because all microstructures have to have the same sets of features
# But grain 1 of microstructure X is not physically the same as grain 1 of microstructure Y.
# This means a problem basically since our data would only have 1 feature : average c per grain.
# It is generally impossible to apply ML to classify 1 feature data.
# 3 solutions,
# solution 1 : We assume C per grain of each grain is a feature instead of a sample => 10 samples 260 features
# This may be bad because we have 10 labels, i.e only 1 sample per label. Try it first to see if its bad

# Solution 2 : somehow expand data dimension. Add more features to your data
# (maybe average Ct, Cl, Vol, or average C under diff conditions)

# Solution 3 : pre-labelling our 10 microstructure with defined labels
# (these 3 micros are bad HE, these 3 micros are good HE, etc).
# The problem with solution 3 is that you need to manually clearly define these labels and classify the micros
ct2scale = np.loadtxt('ct2scale.txt')[:, 1]
ct2scale = ct2scale.reshape(1,-1)
cl2scale = np.loadtxt('cl2scale.txt')[:, 1]
cl2scale = cl2scale.reshape(1,-1)
vol2scale = np.loadtxt('vol2scale.txt')[:, 1]
vol2scale = vol2scale.reshape(1,-1)

ct09 = np.loadtxt('ct09.txt')[:, 1]
ct09 = ct09.reshape(1,-1)
cl09 = np.loadtxt('cl09.txt')[:, 1]
cl09 = cl09.reshape(1,-1)
vol09 = np.loadtxt('vol09.txt')[:, 1]
vol09 = vol2scale.reshape(1,-1)

ct20 = np.loadtxt('ct20.txt')[:, 1]
ct20 = ct20.reshape(1,-1)
cl20 = np.loadtxt('cl20.txt')[:, 1]
cl20 = cl20.reshape(1,-1)
vol20 = np.loadtxt('vol20.txt')[:, 1]
vol20 = vol20.reshape(1,-1)

ctrat = np.loadtxt('ct_ratio.txt')[:, 1]
ctrat = ctrat.reshape(1,-1)
clrat = np.loadtxt('cl_ratio.txt')[:, 1]
clrat = clrat.reshape(1,-1)
volrat = np.loadtxt('vol_ratio.txt')[:, 1]
volrat = volrat.reshape(1,-1)

ctvary = np.loadtxt('ct_spherevary.txt')[:, 1]
ctvary = ctvary.reshape(1,-1)
clvary = np.loadtxt('cl_spherevary.txt')[:, 1]
clvary = clvary.reshape(1,-1)
volvary = np.loadtxt('vol_vary.txt')[:, 1]
volvary = volvary.reshape(1,-1)

ctgg = np.loadtxt('ctgg.txt')[:, 1]
ctgg = ctgg.reshape(1,-1)
clgg = np.loadtxt('clgg.txt')[:, 1]
clgg = clgg.reshape(1,-1)
volgg = np.loadtxt('volgg.txt')[:, 1]
volgg = volgg.reshape(1,-1)

ctirreg = np.loadtxt('ctirreg.txt')[:, 1]  #Shape (261,)
ctirreg = ctirreg.reshape(-1,1)            #Data that needs to be imputed has different reshape
clirreg = np.loadtxt('clirreg.txt')[:, 1]  #Shape (261,)
clirreg = clirreg.reshape(-1,1)
volirreg = np.loadtxt('volirreg.txt')[:, 1]  #Shape (261,)
volirreg = volirreg.reshape(-1,1)

ctn63 = np.loadtxt('ct63lam.txt')[:, 1]      #Shape (264,)
ctn63 = ctn63.reshape(-1,1)                #Data that needs to be imputed has different reshape
cln63 = np.loadtxt('cl63lam.txt')[:, 1]      #Shape (264,)
cln63 = cln63.reshape(-1,1)                #Data that needs to be imputed has different reshape
vol63 = np.loadtxt('vol63lam.txt')[:, 1]      #Shape (264,)
vol63 = vol63.reshape(-1,1)                #Data that needs to be imputed has different reshape

ctparacol = np.loadtxt('ctparacol.txt')[:, 1] #Has nan
ctparacol = ctparacol.reshape(-1,1)        #Data that needs to be imputed has different reshape
clparacol = np.loadtxt('clparacol.txt')[:, 1] #Has nan
clparacol = clparacol.reshape(-1,1)        #Data that needs to be imputed has different reshape
volparacol = np.loadtxt('volparacolumn.txt')[:, 1] #Has nan
volparacol = volparacol.reshape(-1,1)        #Data that needs to be imputed has different reshape

ctvoro = np.loadtxt('ctvoro.txt')[:, 1]
ctvoro = ctvoro.reshape(1,-1)
clvoro = np.loadtxt('clvoro.txt')[:, 1]
clvoro = clvoro.reshape(1,-1)
volvoro = np.loadtxt('volvoro.txt')[:, 1]
volvoro = volvoro.reshape(1,-1)

ctZcom = np.loadtxt('ctZcolumn.txt')[:, 1] #Has nan
ctZcom = ctZcom.reshape(-1, 1)               #Data that needs to be imputed has different reshape
clZcom = np.loadtxt('clZcolumn.txt')[:, 1] #Has nan
clZcom = clZcom.reshape(-1, 1)               #Data that needs to be imputed has different reshape
volZcom = np.loadtxt('volZcolumn.txt')[:, 1] #Has nan
volZcom = volZcom.reshape(-1, 1)               #Data that needs to be imputed has different reshape

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
volirreg = np.delete(volirreg, volirregdrop)

ctn63 = np.delete(ctn63, ctn63drop)
cln63 = np.delete(cln63, cln63drop)
vol63 = np.delete(vol63, voln63drop)

#Reshape the imputed data back to the correct shape for grain = sample
ctirreg = ctirreg.reshape(1,-1)
clirreg = clirreg.reshape(1,-1)
volirreg = volirreg.reshape(1,-1)

ctn63 = ctn63.reshape(1,-1)
cln63 = cln63.reshape(1,-1)
vol63 = vol63.reshape(1,-1)

ctZcom = ctZcom.reshape(1,-1)
clZcom = clZcom.reshape(1,-1)
volZcom = volZcom.reshape(1,-1)

ctparacol = ctparacol.reshape(1,-1) #test
clparacol = clparacol.reshape(1,-1) #test
volparacol = volparacol.reshape(1,-1) #test

#Block to check imputed value. Default commented out
#a = []
#b = []
#c = []
#d = []
#for index in cZcomnan:
#    a.append([cZcom[index-1], cZcom[index], cZcom[index+1]])
#for index in cparacolnan:
#    b.append([cparacol[index-1], cparacol[index], cparacol[index+1]])
#for index in cirregnan:
#    c.append([cirreg[index-1], cirreg[index], cirreg[index+1]])
#for index in cn63nan:
#    d.append([cn63[index-1], cn63[index], cn63[index+1]])
#print(a)
#print(b)
#print(c)
#print(d)


####### Creating training and test data #######
#arrays = [c2scale, c20, cirreg, cZcom, c09, cn63, cgg, cvoro, crat, cvary, cparacol]
allct = np.concatenate([ct2scale.T, ct20.T, ctirreg.T, ctZcom.T, ct09.T, ctn63.T, ctgg.T, ctvoro.T, ctrat.T,ctvary.T])
allcl = np.concatenate([cl2scale.T, cl20.T, clirreg.T, clZcom.T, cl09.T, cln63.T, clgg.T, clvoro.T, clrat.T,clvary.T])
allvol = np.concatenate([vol2scale.T, vol20.T, volirreg.T, volZcom.T, vol09.T, vol63.T, volgg.T, volvoro.T, volrat.T,volvary.T])
alldata = np.concatenate([allct,allcl, allvol], axis = 1)

df = pd.DataFrame(data = alldata)
structures = ['2scale', 'sphere20', 'irregular laminar', 'columnar Z', 'equiaxed',
         'lamellar', 'graingrowth', 'voronoi', 'aspect ratio', 'varying spherecity']
y_train = []
for label in structures:
     for i in range(1,261):
         y_train.append(label)

test = np.concatenate([ctparacol.T, clparacol.T, volparacol.T], axis = 1)


################ Model Evaluation ##############################
# get a list of models to evaluate
def get_models():
    models = dict()
    for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        # create name for model
        key = '%.4f' % p
        # turn off penalty in some cases
        if p == 0.0:
            # no penalty in this case
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
        else:
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)
    return models
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model and collect the scores
    scores = evaluate_model(model, df, y_train)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize progress along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

############# Model Testing ##########################
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
log_reg.fit(df, y_train)

label_pred = log_reg.predict(test)

unique, counts = np.unique(label_pred, return_counts=True)

occur = dict(zip(unique, counts))
for key, value in occur.items():
    value = value*100/260
    occur[key] = value
print(occur)


