## Resources Used:

- Jupyter Notebook 6.1.3 
- Python 3.6

## File Descriptions:

- [finding_donors.ipynb](https://github.com/lizgarseeyah/Finding-Donors/blob/master/finding_donors.ipynb) - main file
-  p1_charityml - zip folder containing csv data sources for the program.

## Overview:

This project applies and evaluates three types of supervised learning models, Ensemble, K-NN, and SVM, to identify potential donors to target and account for how much mailing resources to allocate. Each model is evaluated and scored for accuracy.

## Problem Statement:

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. 

With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. Your goal will be evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent.

## Solution: 

The steps below is a high-level summary of the steps I took to address the problem statement. For a more detailed explaination, please see the [finding_donors.ipynb](https://github.com/lizgarseeyah/Finding-Donors/blob/master/finding_donors.ipynb) file in the GitHub repository.

1. The first part of this program imports and preprocesses the data. Before preprocessing, one must select a feature set that addresses the problem statment. The next step normalizes the data by handling missing, invalid, or outlying data points either removing or performing a method called One-Hot encoding. The table below shows how one-hot encoding works: the function takes a categorical data type and changes to a numerical data type to match the rest of the data.


![one-hot-encoding](/img/One-Hot-encoding.png) 


2. The second part is to split the data into a training set (80% of the data) and a testing set (20% of the data).

```markdown
# Import train_test_split
from sklearn.cross_validation import train_test_split

```

