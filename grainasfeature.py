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
c2scale = np.loadtxt('c2scale.txt')[:, 1]
c2scale = c2scale.reshape(1,-1)
c09 = np.loadtxt('c09.txt')[:, 1]
c09 = c09.reshape(1,-1)
c20 = np.loadtxt('c20.txt')[:, 1]
c20 = c20.reshape(1,-1)
crat = np.loadtxt('c_ratio.txt')[:, 1]
crat = crat.reshape(1,-1)
cvary = np.loadtxt('c_spherevary.txt')[:, 1]
cvary = cvary.reshape(1,-1)
cgg = np.loadtxt('cgg.txt')[:, 1]
cgg = cgg.reshape(1,-1)
cirreg = np.loadtxt('cirreg.txt')[:, 1]  #Shape (261,)
cirreg = cirreg.reshape(-1,1)            #Data that needs to be imputed has different reshape
cn63 = np.loadtxt('cn63.txt')[:, 1]      #Shape (264,)
cn63 = cn63.reshape(-1,1)                #Data that needs to be imputed has different reshape
cparacol = np.loadtxt('cparacol.txt')[:, 1] #Has nan
cparacol = cparacol.reshape(-1,1)        #Data that needs to be imputed has different reshape
cvoro = np.loadtxt('cvronoi.txt')[:, 1]
cvoro = cvoro.reshape(1,-1)
cZcom = np.loadtxt('cZcolumnar.txt')[:, 1] #Has nan
cZcom = cZcom.reshape(-1, 1)               #Data that needs to be imputed has different reshape

################ Dealing with NaN value ####################
cirregnan = np.argwhere(np.isnan(cirreg))[:,0]
cn63nan = np.argwhere(np.isnan(cn63))[:,0]
cZcomnan = np.argwhere(np.isnan(cZcom))[:,0]
cparacolnan = np.argwhere(np.isnan(cparacol))[:,0]

# Defining imputer model
imp_tree = IterativeImputer(random_state=0, missing_values= np.nan,
                            estimator=ExtraTreesRegressor(max_features="sqrt", random_state=0))
cirreg = imp_tree.fit_transform(cirreg)
cn63 = imp_tree.fit_transform(cn63)
cZcom = imp_tree.fit_transform(cZcom)
cparacol = imp_tree.fit_transform(cparacol)

cn63drop = [cn63nan[1],cn63nan[2], cn63nan[3], cn63nan[4]]
cirregdrop = [cirregnan[1]]

cirreg = np.delete(cirreg, cirregdrop)
cn63 = np.delete(cn63, cn63drop)
#Reshape the imputed data back to the correct shape for grain = feature
cirreg = cirreg.reshape(1,-1)
cn63 = cn63.reshape(1,-1)
cZcom = cZcom.reshape(1,-1)
cparacol = cparacol.reshape(1,-1) #test


# region Description:  Block to check imputed values (default commented out)
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

#region end

####### Creating training data #######
arrays = [c2scale, c20, cirreg, cZcom, c09, cn63, cgg, cvoro, crat, cvary, cparacol]
alldata = np.concatenate([c2scale.T, c20.T, cirreg.T, cZcom.T, c09.T, cn63.T, cgg.T, cvoro.T, crat.T, cvary.T])
df = pd.DataFrame(data = alldata)
structures = ['2scale', 'sphere20', 'irregular laminar', 'columnar Z', 'equiaxed',
         'lamellar', 'graingrowth', 'voronoi', 'aspect ratio', 'varying spherecity']
y_train = []
for label in structures:
     for i in range(1,261):
         y_train.append(label)

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

label_pred = log_reg.predict(cparacol.T)

unique, counts = np.unique(label_pred, return_counts=True)

occur = dict(zip(unique, counts))
for key, value in occur.items():
    value = value*100/260
    occur[key] = value
print(occur)


