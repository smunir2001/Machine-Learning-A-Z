smunir2001@gmail.com | November 9th, 2022 | ReadMe.md
# Machine Learning for Everybody - YouTube | freecodecamp.org
## Introduction/overview
Each row in the dataset is a __sample__ (an item of the dataset; a datapoint).

The column headers are qualities (__labels/features__).

Column headers are known as __features__.

The ouput is known as the __target__ for that specific feature vector.

__Features matrix__ X & __Labels/targets vector__ Y
* we pass features to our ML model to predict the label (class column)
* supervised learning
## What is machine learning?
Machine learning is a subdomain of computer science that focuses on algorithms which help a computer learn from data without explicit programming.
## AI vs ML vs DS
__Arificial intelligence__ is an area of computer science, where the goal is to enable computers and machines to perform human-like taks and simulate human behavior.

__Machine learning__ is a subset of Ai that tries to solve a specific problem and make predictino using data.

__Data science__ is a field that attempts to find patterns and draw insights from data (might use ML!); all fields overlap.
## Types of Machine Learning
1. __Supervised learning -__ uses labeled inputs (meaning the input has a corresponding output label) to train models and learn outputs.
2. __Unsupervised learning -__ uses unlabeled data to learn about patterns in data.
3. __Reinforcement learning -__ agent learning in interactive environment based on rewards and penalties.
## Supervised Learning
inputs -> model -> output (prediction)

All the inputs (features) are referred to as the __feature vector__.
### Features
* Qualitative - categorical data (finite number of categories or groups).
    * Nominal data (no inherent order)
        * ONE-HOT ENCODING
* Quantitative - numerical valued data (could be discrete or continuous).
## Types of Predictions
### Supervised learning taks
1. __Classification -__ predict discrete classes
    * multi-class classification
    * binary classification
2. __Regression -__ predicit continuous values
## Supervised Learning Dataset
* Training dataset
* Validation dataset
* Testing dataset
1. Feed training dataset into the model -> __loss__ is known as the difference between our predictions and the true values.
    * make adjustments (known as __training__)
    * The validation set is used as a reality check during/after training to ensure the model can handle unseen data.
    * The test set is used to check how generalizable the final chosen model is.
## Metrics of Performance
__Loss:__ the difference between your prediction and the actual label.
* L1 loss = sum(|yReal -yPredicted|) -> absolute value function
* L2 loss = sum((yReal - yPredicted)^2) -> quadratic value function
* Binary Cross-Entropy Loss
    * loss = -1/N * sum(yReal * log(yPredicted) + (1 - yReal) * log((1 - yPredicted)))
        * loss decreases as the performance gets better.

__Accuracy:__ predictions/actual
## K-Nearest Neighbors (K-NN)
Look at what's around you, and take the label of the majority that's around me.
* Define a distance function (Euclidean distance)
    * straight line distnce from current point to nearest label (sample).
    ```
    d = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    ```
    * __k__ tells us how many neighbors to use to judge the label.
        * all depends on how big the data is
## Naive Bayes
MAP (Maximum A Posteriori)
* what is the probility that something is from a specific class
    * posterior
    * liklihood
    * prior
## Logistic Regression
* Sigmoid Function
## Support Vector Machines (SVM)
* margins (goal is to maximize margin)
* find straight line to distictively split the two classes
* not robust with outliers