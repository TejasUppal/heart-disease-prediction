import numpy as np # np is short for numpy

import pandas as pd # pandas is so commonly used, it's shortened to pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay 
df = pd.read_csv("heart-disease.csv")
df.shape
df.head()
df.target.value_counts()
df.target.value_counts(normalize=True)
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);
df.info()
df.describe()
df.sex.value_counts()
pd.crosstab(index=df.target, columns=df.sex)
pd.crosstab(df.target, df.sex).plot(kind="bar",  figsize=(10,6),  color=["salmon", "lightblue"]);
                                   
                                  
pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"])
plt.title("Heart Disease Frequency vs Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0);

plt.figure(figsize=(10,6))

plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="salmon") # define it as a scatter figure
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="lightblue") # axis always come as (x, y)
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate");
df.age.plot.hist();
pd.crosstab(index=df.cp, columns=df.target)

pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(10,6),  color=["lightblue", "salmon"])
                                  
                                   
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation = 0);
corr_matrix = df.corr()
corr_matrix
corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");
X = df.drop(labels="target", axis=1)

y = df.target.to_numpy()
X.head()
y, type(y)
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
X_train.head()
y_train, len(y_train)
X_test.head()
y_test, len(y_test)
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(max_iter=100), # Note: if you see a warning about "convergence not reached", you can increase `max_iter` until convergence is reached
          "Random Forest": RandomForestClassifier()}

def fit_and_score(models, X_train, X_test, y_train, y_test):
   np.random.seed(42)
   
    model_scores = {}
 
for name, model in models.items():
model.fit(X_train, y_train)
model_scores[name] = model.score(X_test, y_test)
return model_scores
model_scores = fit_and_score(models=models, X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)
model_scores
model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();
train_scores = []


test_scores = []
neighbors = range(1, 21) # 1 to 20
knn = KNeighborsClassifier()
for i in neighbors:
    knn.set_params(n_neighbors = i) # set neighbors value
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
train_scores  
plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()
print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
%%time

np.random.seed(42)
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)
rs_log_reg.fit(X_train, y_train);
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)
%%time 


np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)


rs_rf.fit(X_train, y_train);
Fitting 5 folds for each of 20 candidates, totalling 100 fits
CPU times: user 21.6 s, sys: 144 ms, total: 21.8 s
Wall time: 22.1 s

rs_rf.best_params_
{'n_estimators': np.int64(210),
 'min_samples_split': np.int64(4),
 'min_samples_leaf': np.int64(19),
 'max_depth': 3}

rs_rf.score(X_test, y_test)
%%time
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

gs_log_reg.fit(X_train, y_train);
gs_log_reg.best_params_
gs_log_reg.score(X_test, y_test)
                                                   
