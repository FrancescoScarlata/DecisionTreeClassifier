## Weka instructions (on simple cli):

c45 tree with no test set. (this will just specify the training file).
This will just make a tree classifier and will do some cross-validation on the dataset.

'''java weka.classifiers.trees.J48 -t path/dataset.arff'''
