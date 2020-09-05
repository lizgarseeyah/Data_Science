## Project Overview

### Resources Used:

- Jupyter Notebook 6.1.3 
- Python 3.6

### File Descriptions:

- [finding_donors.ipynb](https://github.com/lizgarseeyah/Finding-Donors/blob/master/finding_donors.ipynb) - main file
-  p1_charityml - zip folder containing csv data sources for the program.

### Summary:

This project applies and evaluates three types of supervised learning models, Ensemble, K-NN, and SVM, to identify potential donors to target and account for how much mailing resources to allocate. Each model is evaluated and scored for accuracy.

## Problem Statement:

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. 

With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. Your goal will be evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent.

## Solution: 

The steps below is a high-level summary of the steps I took to address the problem statement. For a more detailed explaination, please see the [finding_donors.ipynb](https://github.com/lizgarseeyah/Finding-Donors/blob/master/finding_donors.ipynb) file in the GitHub repository.

### Data Exploration and preprocessing

The first part of this program imports and preprocesses the data. Before preprocessing, one must select a feature set that addresses the problem statment. The next step normalizes the data by handling missing, invalid, or outlying data points either removing or performing a method called **one-hot encoding**. The table below shows how one-hot encoding works: the function takes a categorical data type and changes to a numerical data type to match the rest of the data.


![one-hot-encoding](/img/One-Hot-encoding.png) 

### Train and test the data using a supervised learning method 

The first step before using the training models is to shuffle and split the data into a training set (80% of the data) and a testing set (20% of the data).

```markdown
# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
```
The training set will be used to train the data on three supervised learning models: AdaBoost Ensemble, K-Nearest Neighbors (KNN), and Support Vecotr Machine (SVM).

```markdown
# Import the three supervised learning models from sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

## Initialize the three models
clf_A = SVC(random_state=1)#SVC
clf_B = KNeighborsClassifier()
clf_C = AdaBoostClassifier(random_state=1)#Ensemble Methods
```
An initial model evaluation was performed by running 1%, 10%, and 100% of the training data through each model. The accuracy and f-score metrics are used to determine the best model. Accuracy is a relevant metric, because we do not want to request donations from a person that we incorrectly identified as a potential donor. The f-score measure both precision, true positives, and the models ability to recall those individuals.

![accuracy-f-score](/img/accuracy-f-score.png) 
![performance-metrics](/img/performance-metrics.png) 

After completing the evaluation, the testing set results above indicated that the AdaBoost Classifier was the best model for this project.

**AdaBoost Overview**
Adaboost belongs to the family of ensemble methods. Ensemble means we take into account a set of multiple "weak" hypothesis and combine them to form one "strong" hypothesis. At each iteration, a "weak" hypothesis attempts to classify the training data, here for example it tries to approximately find the individuals making more than fifty-thousand dollars All the misclassified individuals are more heavily weighted and more focus will be given to the hard-to-classify points in the attempt to classify them correctly at the next iteration. Iteration after iteration, the combination of all those "weak" learners should converge towards a more confident, stronger hypothesis allowing to find individuals making more than fifty-thousand dollars. 

The only requirement for this model to converge well is that every "weak" learner need to be slightly better than random guessing. An analogy to this method would be to ask a crowd of unexperienced doctors to diagnose a disease rather than asking only one expert. The expert would more often be right than each doctor taken individually, but when considering the crowd, all the answers should converge towards the right outcome with more and more confidence as we ask more doctors (given that each unexperienced doctor is still doing better than a randomly guessing, that the previous doctor communicates the results of his analysis to the next one and that each doctor focuses on what the previous one wasn't able to find).


