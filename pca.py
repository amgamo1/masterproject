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

#Random informative CP numbers
n_info = randint(90,300)
print(n_info)
data_features = randint(1000,2000)
redun = data_features - n_info

# define dataset
X, y = make_classification(n_samples=1000,
                           n_classes = 11,
                           n_features=data_features,
                           n_informative=n_info, n_redundant= redun, random_state=7)
print(X.shape)
#Training test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#PCA
pca = PCA().fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
##Cumvar has as many datapoints as n_samples
samples = list(range(1,1001))
plt.ylim(0.0,1.1)
plt.plot(samples, cumvar, linestyle='--', color='b')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

#relevant component
n_rele= np.count_nonzero(cumvar<=0.99)
print(n_rele)
pca_actual = PCA(n_components = n_rele)
X_trainpca  = pca_actual.fit_transform(X_train)
X_testpca = pca_actual.fit_transform(X_test)
#print(X_trainpca.shape)
#print(X_testpca.shape)
#X_pca = pca_actual.fit_transform(X)


#Model
model = LogisticRegression(multi_class='ovr', solver='lbfgs')
model.fit(X_trainpca, y_train)
prediction=model.predict(X_testpca)
print(model.score(X_testpca, y_test))
print(metrics.classification_report(y_test, model.predict(X_testpca)))

#Creating matplotlib axes object to assign figuresize and figure title
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Confusion Matrx')

disp =metrics.plot_confusion_matrix(model, X_testpca, y_test, display_labels= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ax = ax)
disp.confusion_matrix
plt.show()
#print(prediction[100])
#print(y_test[100])


#Logistic regression
# define the pipeline
#steps = [('pca', PCA(n_components=n_rele)), ('m', LogisticRegression())]
#model = Pipeline(steps=steps)
# evaluate model
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


