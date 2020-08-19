#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import *
import matplotlib.pyplot
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from time import time 
import numpy as np
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

#import sklearn.tree import DecisionTreeClassifier()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',  
                 'exercised_stock_options', 
                 'bonus', 
                 'to_messages', 
                 'from_poi_to_this_person', 
                 'from_messages', 
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees'] # You will need to use more features
'''feture_list options:  ['salary',
                          'deferral_payments',
                          'total_payments',
                          'loan_advances',
                          'bonus',
                          'restricted_stock_deferred',
                          'deferred_income',
                          'total_stock_value',
                          'expenses',
                          'exercised_stock_options',
                          'other',
                          'long_term_incentive',
                          'restricted_stock',
                          'director_fees']
        email features: ['to_messages',
                         'email_address',
                         'from_poi_to_this_person',
                         'from_messages',
                         'from_this_person_to_poi',
                         'shared_receipt_with_poi'] '''

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

            
    
    

### Task 2: Remove outliers

'''def Vfeatures(data_dict, f_x, f_y):
    data = featureFormat(data_dict, [f_x, f_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )

    matplotlib.pyplot.xlabel(f_x)
    matplotlib.pyplot.ylabel(f_y)
    matplotlib.pyplot.show()
#plotting
#print data_dict
Vfeatures(data_dict, 'salary', 'bonus')
Vfeatures(data_dict, 'total_stock_value', 'exercised_stock_options')'''

#Total column is an outlier - removing this outlier
data_dict.pop( 'TOTAL', 0 ) 

'''Vfeatures(data_dict, 'salary', 'bonus')
Vfeatures(data_dict, 'total_stock_value', 'exercised_stock_options')'''

### Task 3: Create new feature(s)



def computeFraction( poi_messages, all_messages ):
    fraction = 0.
    if poi_messages == 'NaN':
        return fraction
    elif all_messages == 'NaN':
        return fraction
    else:   
        fraction = float(poi_messages)/float(all_messages)
    return fraction

for name in data_dict:

    data_point = data_dict[name]
    
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_dict[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    
    data_dict[name]["fraction_to_poi"] = fraction_to_poi
    
 ### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list=features_list+['fraction_to_poi',
                             'fraction_from_poi']
'''clf = tree.DecisionTreeClassifier()
fi = clf.feature_importances_
indices = np.argsort(fi)[::-1]
print("Feature ranking:")
for i in range(len(indices)):
    print("%d. feature %s (%f)" % (i+1, features_list[1:][indices[i]], fi[indices[i]]))'''




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

kbest = SelectKBest(f_classif)
k_best = kbest.fit(features, labels)
results  = zip(features_list[1:], k_best.scores_)
results = sorted(results, key=lambda x: x[1], reverse=True)
print "Select K Best Results:"
for a,b in enumerate(results, 1):
    print "{}.{}".format(a,b)

features_list =['poi',
                'exercised_stock_options',
                'total_stock_value',
                'bonus',
                'salary',
                'fraction_to_poi',
                'deferred_income',
                'long_term_incentive']
pipe = Pipeline([('kbest', kbest), ('dt', GaussianNB())])
#parameters = {'criterion':['entropy','gini']}

grid = GridSearchCV(pipe, {'kbest__k': [1,2,3,4]})

clf_fit = grid.fit(features, labels)
clf=clf_fit.best_estimator_
'''#test_classifier(clf, data_dict, features_list)
pipe = Pipeline([('kbest', kbest), ('dt', tree.DecisionTreeClassifier())])
#parameters = {'criterion':['entropy','gini']}

grid = GridSearchCV(pipe, {'kbest__k': [1,2,3,4]})

clf_fit = grid.fit(features, labels)
clf=clf_fit.best_estimator_

print features_list'''

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.

clf = GaussianNB()
#fitting
t0 = time()
clf.fit(features, labels)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
test_classifier(clf, my_dataset, features_list)
print "testing time:", round(time()-t0, 3), "s"
from sklearn import svm

clf = Pipeline([('scaler', StandardScaler()), ('svc', svm.SVC(kernel = 'rbf', C=10000.0))])
#clf = svm.SVC(kernel = 'rbf', C=10000.0)

t0 = time()
clf.fit(features, labels)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
test_classifier(clf, my_dataset, features_list)
print "testing time:", round(time()-t0, 3), "s"

clf = tree.DecisionTreeClassifier()
t0 = time()
clf.fit(features, labels)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
test_classifier(clf, my_dataset, features_list)
print "testing time:", round(time()-t0, 3), "s"



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html






# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
sss.get_n_splits(features, labels)
print sss

#best estimator restults
'''DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=1,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='random')'''
#GridSearchCV
parameters = {}
clf = GridSearchCV(GaussianNB(), parameters, cv=sss).fit(features_train, labels_train)
print 'GridSearchCV', clf.best_estimator_
clf.fit(features, labels)
test_classifier(clf, data_dict, features_list)

'''
#GridSearchCV
parameters = {}
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=sss).fit(features_train, labels_train)
print 'GridSearchCV', clf.best_estimator_
clf.fit(features, labels)
test_classifier(clf, data_dict, features_list)
#MinMaxScalar
mms=MinMaxScaler()
mms.fit(features_train)
features_train=mms.transform(features_train)
print 'MinMaxScaler', test_classifier(clf, data_dict, features_list)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
clf = AdaBoostClassifier((DecisionTreeClassifier(), parameters))
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print confusion_matrix(labels_test, pred)


test_classifier(clf, data_dict, features_list)
'''




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

