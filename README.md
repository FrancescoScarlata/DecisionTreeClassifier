# Decision Tree Classifier

## Introdution
This is a repository for a university project. 

The subject is "Metodi statistici per l'apprendimento" and we want to do a tree classifier.

Thanks to @random-forests for the tutorial on developing a decision tree classifier from scratch. Part of scripts are made by him in a youtube tutorial.

You can see in the history what has be done by me.


## Dependencies
Most of the modules are built in, but you can install the remaining ones with: ```pip install x```

These are the modules used in this project:
- csv
- math
- numpy
- pathlib
- time
- os.path
- matplotlib
- argparse

## Dataset setup

1. put the dataset data (.csv) in  "Datasets"
2. put the dataset header (.csv) in  "Datasets". Just the titles of the columns


## How to use the scripts
(These instructions are for windows)
1. Go in the Source folder and use  '''python DecisionTreeClassifier.py'''. If you want to see the resulting tree, use -d at the end: '''python DecisionTreeClassifier.py -d'''
2. Write the relative Path inside the "Datasets" folder for the dataset data. For example, to use the dataset 'car_data.csv' inside the 'CarEvaluation' folder, we'll write "CarEvaluation\car_data.csv".
3. Write the relative Path inside the "Datasets" folder for the header of the dataset. For example, to use the header 'car_header.csv' inside the 'CarEvaluation' folder, we'll write "CarEvaluation\header_data.csv".
4. Write the number of the indixes that have a numeric value. This is to distinguish the cases '<=' to '=='. In case there are no numeric columns, just use the 'Enter' key.

If it doesn't do exception, good job, you just need to wait :D 

## How to use WEKA

1. Open Weka (assuming it is installed)
2. Go to "simple CLI"
3. Use the instruction in the "Weka Instruction" file.
Note: the path should be the absolute file of the dataset. Use the .arff file. The data are the same, but there are written also the attributes.

