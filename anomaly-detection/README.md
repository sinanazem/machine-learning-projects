# Outlier Detection
<img src="https://images.ctfassets.net/kj4bmrik9d6o/cESitsCxzL2ijivbNwiW6/de9421d4c144e5a5c23c35941931c83f/Outlier_Graph_CalculateOutlierFormula-01.png">

Outlier Detection Python is a specialized task which has various use-cases in Machine
Learning. Use-cases would be anomaly detection, fraud detection, outlier detection etc. There
are many ways we can find outliers in your analysis.


## What are outliers?
Outliers: in simple terms outliers are data points which are significantly different from your
entire datasets.

## How do outliers occur in a datasets?

Outliers occur either by chance, or either by measurement error or data population is heavy
tailed distribution as shown above.
Main effects of having outliers are that they can skew your analytics in poor analysis, longer
training time and bad results at the end. Most importantly, this distorts the reality which
exists in the data.

## Simple methods to Identify outliers in your datasets.

**Sorting** 
<br>If you have dataset you can quickly just sort ascending or descending.
While it is looks so obvious, but sorting actually works on real world.<br>
**Outlier Detection Python** 
<br>Quick Method in Pandas – Describe( )

```
import numpy as np
import pandas as pd
url =
'https://raw.githubusercontent.com/Sketchjar/MachineLearningHD'
df = pd.read_csv(url)
df.describe()
```

## Outlier Detection Using Machine Learning
Robust Covariance – Elliptic Envelope<br>
This method is based on premises that outliers in a data leads increase in covariance, making
the range of data larger. Subsequently the determinant of covariance will also increase, this
in theory should reduce by removing the outliers in the datasets. This method assumes that
some of hyper parameters in n samples follow Gaussian distribution. Here is flow on how
this works:
<br>
 #### One-Class SVM
One class Support Vector Machine is a special case in support vector machines which is used
for unsupervised outlier detection. For more information on support vector

Let see outlier detection python code using One Class SVM. We will see two different
examples for it.
```
from sklearn.svm import OneClassSVM
X = [[0], [0.44], [0.45], [0.46], [1]]
clf = OneClassSVM(gamma='auto').fit(X)
clf.predict(X)
array([-1, 1, 1, 1, -1, -1, -1], dtype=int64)
```
<br>output:
```
array([-1, 1, 1, 1, -1, -1, -1], dtype=int64)
```
Here -1 refers to outlier and 1 refers to not an outliers.
<br>
Let us see another example:
```
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm


xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Outlier Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)

a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
edgecolors='k')

c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
edgecolors='k')
plt.axis('tight')

plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
["learned frontier", "training observations",
"new regular observations", "new abnormal observations"],
loc="upper left",
prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
"error train: %d/200 ; errors novel regular: %d/40 ; "
"errors novel abnormal: %d/40"
% (n_error_train, n_error_test, n_error_outliers))
plt.show()


```


#### Isolation Forest
Isolation Forest is an ensemble model which isolates observations by randomly selecting a
feature and selecting a split value between maximum and minimum of selected feature.<br>
Since this recursive partitioning is represented by a tree structure, and number of splittings is
equivalent to path length from root node to terminating node.
<br><br>

<img src="https://miro.medium.com/max/1400/1*6GFMewU1Aax57nsW2uSakQ.png" high=450 width=550>

See Isolation Forest in code.
```
from sklearn.ensemble import IsolationForest
X = [[-1.1], [0.3], [0.5], [100]]
clf = IsolationForest(random_state=0).fit(X)
clf.predict([[0.1], [0], [90]])
```
<br>output:
```
array([ 1, 1, -1])
```
Here -1 refers to outlier and 1 refers to not an outliers.

#### Local Outlier Factor (LOF)
LOF computes local density deviation of a certain point as compared to its neighbors. It is
different variant of k Nearest neighbors. Simply, in LOF outliers is considered to be points
which have lower density than its neighbors.

Local Outlier Factor in Code:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


print(__doc__)
np.random.seed(42)

# Generate train data
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# Generate some outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]
n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_
plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')

# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()

```
In Summary , we have discussed various quick methods through we can identify outliers.
There are other advanced machine learning models which can also be used to identify
outliers
