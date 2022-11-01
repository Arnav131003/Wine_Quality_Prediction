
# Wine_Quality_Prediction 🍷 
The aim of this project is to understand different types of learning algorithms on a popular wine quality dataset on kaggle using machine learning.

## Libraries Used
- ### Numpy 
     <a href="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fres.cloudinary.com%2Fpracticaldev%2Fimage%2Ffetch%2Fs--IS2P_PRA--%2Fc_imagga_scale%2Cf_auto%2Cfl_progressive%2Ch_420%2Cq_auto%2Cw_1000%2Fhttps%3A%2F%2Fres.cloudinary.com%2Fpracticaldev%2Fimage%2Ffetch%2Fs--PmX0XWGn--%2Fc_imagga_scale%252Cf_auto%252Cfl_progressive%252Ch_420%252Cq_auto%252Cw_1000%2Fhttps%3A%2F%2Fthepracticaldev.s3.amazonaws.com%2Fi%2Fi7xbfqoej9ylzboevtbb.png"><img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fres.cloudinary.com%2Fpracticaldev%2Fimage%2Ffetch%2Fs--IS2P_PRA--%2Fc_imagga_scale%2Cf_auto%2Cfl_progressive%2Ch_420%2Cq_auto%2Cw_1000%2Fhttps%3A%2F%2Fres.cloudinary.com%2Fpracticaldev%2Fimage%2Ffetch%2Fs--PmX0XWGn--%2Fc_imagga_scale%252Cf_auto%252Cfl_progressive%252Ch_420%252Cq_auto%252Cw_1000%2Fhttps%3A%2F%2Fthepracticaldev.s3.amazonaws.com%2Fi%2Fi7xbfqoej9ylzboevtbb.png" maxheight="25%" maxwidth="25%" ></a>

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
   <a href="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.freecodecamp.org%2Fnews%2Fcontent%2Fimages%2Fsize%2Fw2000%2F2020%2F07%2Fpandas-logo.png"><img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.freecodecamp.org%2Fnews%2Fcontent%2Fimages%2Fsize%2Fw2000%2F2020%2F07%2Fpandas-logo.png" maxheight="25%" maxwidth="25%" ></a>
    #### Importing Pandas Library
    ```bash
    import pandas as pd
    ```
    #### About Pandas
    Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python.

- ### Seaborn
    <a href="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg"><img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" maxheight="25%" maxwidth="25%" ></a>
    #### Importing Seaborn
    ```bash
    import seaborn as sns
    ```
    #### About Seaborn
    Seaborn is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures.It helps you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. 


- ### Matplotlib
    <a href="https://matplotlib.org/_static/images/logo_dark.svg"><img src="https://matplotlib.org/_static/images/logo_dark.svg" maxheight="25%" maxwidth="25%" ></a>
    #### Importing Matplolib
    ```bash
    import matplotlib.pyplot as plt
    ```
    #### About Matplotlib
    Matplotlib is easy to use and an amazing visualizing library in Python. It is built on NumPy arrays and designed to work with the broader SciPy stack and consists of several plots like line, bar, scatter, histogram, etc. 
- ### Sklearn
    <a href="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Famueller.github.io%2Fsklearn_014_015_pydata%2Fsklearn-logo.png"><img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Famueller.github.io%2Fsklearn_014_015_pydata%2Fsklearn-logo.png" maxheight="25%" maxwidth="25%" ></a>
    #### Importing Sklearn
    ```bash
    import sklearn
    ```
    #### About Sklearn
    Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python. This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.


## Algorithms Used 
- ### Logistic Regression
    ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F2312%2F1*iKo3KI4kqkZ47W7pmmH4cw.png&f=1&nofb=1&ipt=b1bb2f75f4cd5dacbdccb79dcb7ec65b4226c2fc670fd04bad174bf0e291892f&ipo=images)
    #### Importing Logistic Regression Classifier
    ```bash
    from sklearn.linear_model import LogisticRegression
    ```
    #### About
    Logistic Regression is an easily interpretable classification technique that gives the probability of an event occurring, not just the predicted classification. It also provides a measure of the significance of the effect of each individual input variable, together with a measure of certainty of the variable's effect.
- ### Decision Tree Classifier
    ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.ytimg.com%2Fvi%2FZVR2Way4nwQ%2Fmaxresdefault.jpg&f=1&nofb=1&ipt=929082246a926e05d618680c9e97211ec3812c74238d5a6c12f09c4d714046f6&ipo=images)
    #### Importing Decision Tree Classifier
    ```bash
    from sklearn.tree import DecisionTreeClassifier
    ```
    #### About
    Decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.
- ### Random Forest Classifier
    ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.freecodecamp.org%2Fnews%2Fcontent%2Fimages%2F2020%2F08%2Fhow-random-forest-classifier-work.PNG&f=1&nofb=1&ipt=2c6e870d452397c91b9731c44b61d8171d1532f3fed96ae39e6086a57d761d09&ipo=images)
    #### Importing Random Forest Classifier
    ```bash
    from sklearn.ensemble import RandomForestClassifier
    ```
    #### About
    Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set. 
- ### Support Vector Machine 
    ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fdataaspirant.com%2Fwp-content%2Fuploads%2F2020%2F12%2F3-Support-Vector-Machine-Algorithm.png&f=1&nofb=1&ipt=3eafb226d7951832bacdd2f2aaee49341862fdb4cc71a4053a2d8096e59e2a3e&ipo=images)
    #### Importing Support Vector Machine CLassifier
    ```bash
    from sklearn import svm
    ```
    #### About
    Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.
- ### KNeighbours Classifier 
    ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.theclickreader.com%2Fwp-content%2Fuploads%2F2020%2F08%2F24-1536x864.png&f=1&nofb=1&ipt=a345408b22c331fa31ebdd83748c6d4ea4b51255b259d7593de5bb03b6fd6875&ipo=images)
    #### Importing KNeighbours Classifier
    ```bash
    from sklearn.neighbors import KNeighborsClassifier
    ```
    #### About
    k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another
- ### Gradient Boosting Classifier
    ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F0*paPv7vXuq4eBHZY7.png&f=1&nofb=1&ipt=965fdd200c3a72634251a8fed8da7e96396ef4122ff2ebe3673205814317bbf4&ipo=images)
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

# Enjoy your wine
![](https://cdn.dribbble.com/users/1241808/screenshots/2833908/winedrib.gif)
