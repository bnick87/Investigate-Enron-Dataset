#Errors with Tester.py

'''Traceback (most recent call last):
  File "C:\Users\Bridget\Downloads\ud120-projects-master (1)\ud120-projects-master\final_project\poi_id.py", line 160, in <module>
    test_classifier(clf, data_dict, features_list)
  File "C:\Users\Bridget\Downloads\ud120-projects-master (1)\ud120-projects-master\final_project\tester.py", line 33, in test_classifier
    for train_idx, test_idx in cv:
TypeError: 'StratifiedShuffleSplit' object is not iterable'''

'''https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
Per the above link that was given by the first review it shows the only parameters are
Parameters
n_splitsint, default=10
Number of re-shuffling & splitting iterations.

test_sizefloat or int, default=None
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.1.

train_sizefloat or int, default=None
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.

random_stateint or RandomState instance, default=None
Controls the randomness of the training and testing indices produced. Pass an int for reproducible output across multiple function calls. See Glossary.

Therefore I have changed Tester.py Accordingly as found on
https://stackoverflow.com/questions/53899066/what-could-be-the-reason-for-typeerror-stratifiedshufflesplit-object-is-not

"[..]
from sklearn.model_selection import StratifiedShuffleSplit
[..]
#cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
[..]
#for train_idx, test_idx in cv:
for train_idx, test_idx in cv.split(features, labels):
[..]"'''

'''Also there is no longer cross_validation available it is now model_selection that holds
StratifiedShuffleSplit and test_train_split

I had to correc this in the tester.py as well'''

