
# Wine_Quality_Prediction
The aim of this project is to understand different types of learning algorithms on a popular wine quality dataset on kaggle using machine learning.

## Libraries Used
- ### Numpy 
    ![Numpy](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fres.cloudinary.com%2Fpracticaldev%2Fimage%2Ffetch%2Fs--IS2P_PRA--%2Fc_imagga_scale%2Cf_auto%2Cfl_progressive%2Ch_420%2Cq_auto%2Cw_1000%2Fhttps%3A%2F%2Fres.cloudinary.com%2Fpracticaldev%2Fimage%2Ffetch%2Fs--PmX0XWGn--%2Fc_imagga_scale%252Cf_auto%252Cfl_progressive%252Ch_420%252Cq_auto%252Cw_1000%2Fhttps%3A%2F%2Fthepracticaldev.s3.amazonaws.com%2Fi%2Fi7xbfqoej9ylzboevtbb.png&f=1&nofb=1&ipt=6cebde7404bdf1f7480aa8043418de8779de6898ed92b838ff31e064b4dd401e&ipo=images)
    #### Importing Numpy Library
    ```bash
     import numpy as np
    ```
    #### About Numpy
    Numpy is a library for the Python programming language, 
    adding support for large, multi-dimensional arrays and matrices, 
    along with a large collection of high-level mathematical functions
    to operate on these arrays.
- ### Pandas
   ![Pandas](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.freecodecamp.org%2Fnews%2Fcontent%2Fimages%2Fsize%2Fw2000%2F2020%2F07%2Fpandas-logo.png&f=1&nofb=1&ipt=1b83ff9f7748e6ff21773acb5c4d2df2a849db6634950aded6460e7cf1cf9803&ipo=images)
    #### Importing Pandas Library
    ```bash
    import pandas as pd
    ```
    #### About Pandas
    Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python.

- ### Seaborn
    ![seaborn](https://seaborn.pydata.org/_static/logo-wide-lightbg.svg)
    #### Importing Seaborn
    ```bash
    import seaborn as sns
    ```
    #### About Seaborn
    Seaborn is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures.It helps you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. 


- ### Matplotlib
    ![](https://matplotlib.org/_static/images/logo_dark.svg)
    #### Importing Matplolib
    ```bash
    import matplotlib.pyplot as plt
    ```
    #### About Matplotlib
    Matplotlib is easy to use and an amazing visualizing library in Python. It is built on NumPy arrays and designed to work with the broader SciPy stack and consists of several plots like line, bar, scatter, histogram, etc. 
- ### Sklearn
    ![](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Famueller.github.io%2Fsklearn_014_015_pydata%2Fsklearn-logo.png&f=1&nofb=1&ipt=5d119b094f8b5ff887823aff320195dfdffdba1a5417691430b71675b572207e&ipo=images)
    #### Importing Sklearn
    ```bash
    import sklearn
    ```
    #### About Sklearn
    Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python. This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.


## Algorithms Used 
- ### Logistic Regression
    #### Importing Logistic Regression Classifier
    ```bash
    from sklearn.linear_model import LogisticRegression
    ```
    #### About
    Logistic Regression is an easily interpretable classification technique that gives the probability of an event occurring, not just the predicted classification. It also provides a measure of the significance of the effect of each individual input variable, together with a measure of certainty of the variable's effect.
- ### Decision Tree Classifier
    #### Importing Decision Tree Classifier
    ```bash
    from sklearn.tree import DecisionTreeClassifier
    ```
    #### About
    Decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.
- ### Random Forest Classifier
    #### Importing Random Forest Classifier
    ```bash
    from sklearn.ensemble import RandomForestClassifier
    ```
    #### About
    Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set. 
- ### Support Vector Machine 
    #### Importing Support Vector Machine CLassifier
    ```bash
    from sklearn import svm
    ```
    #### About
    Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.
- ### KNeighbours Classifier 
    #### Importing KNeighbours Classifier
    ```bash
    from sklearn.neighbors import KNeighborsClassifier
    ```
    #### About
    k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another
- ### Gradient Boosting Classifier
    #### Importing Gradient Boosting Classifier
    ```bash
    from sklearn.ensemble import GradientBoostingClassifier
    ```
    #### About
    Gradient boosting algorithm is one of the most powerful algorithms in the field of machine learning. As we know that the errors in machine learning algorithms are broadly classified into two categories i.e. Bias Error and Variance Error. As gradient boosting is one of the boosting algorithms it is used to minimize bias error of the mode

## Dataset Analysis

- ### Quality_Count Analysis
    ![Count v/s Quality](https://github.com/Arnav131003/Wine_Quality_Prediction/blob/main/%20Quality_Count.png?raw=true)
- ### Alcohol v/s Quality Plot 
    - Using Barplot visualizing the change of quality of wine on the basis of alcohol amout present in it
    ![Alcohol v/s Quality](https://github.com/Arnav131003/Wine_Quality_Prediction/blob/main/Alcohol_Quality.png?raw=true)
- ### HeatMap Analysis of Features
    - Determing the co-relation of different features among each other 
    ![heatmap](https://github.com/Arnav131003/Wine_Quality_Prediction/blob/main/Heat_Map_features.png?raw=true)
- ### Features Pairplot Analysis 
    - Pairplot brings the ability of visualizing all features against each other at the same time 
    ![Alcohol v/s Quality](https://github.com/Arnav131003/Wine_Quality_Prediction/blob/main/Features_pairplot_1.png?raw=true)
    ![Alcohol v/s Quality](https://github.com/Arnav131003/Wine_Quality_Prediction/blob/main/Features_pairplot_2.png?raw=true)
## Model Analysis 
   - Plotting the accuracy of different models used 
   ![Model_Accuracy](https://github.com/Arnav131003/Wine_Quality_Prediction/blob/main/Model_Accuracy.png?raw=true)


