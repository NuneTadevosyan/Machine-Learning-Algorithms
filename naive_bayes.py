import numpy as np
import pandas as pd

class MyNaiveBayes:
  def __init__(self, smoothing=False):
      self.smoothing = smoothing
    
  def fit(self, X_train, y_train):
      self.X_train = X_train
      self.y_train = y_train
      self.priors = self.calculate_priors()
      self.likelihoods = self.calculate_likelihoods()      
      
  def predict(self, X_test):
    post = []
    l = list(self.y_train.unique())
    for i in range(len(X_test)):
      inn = pd.MultiIndex.from_arrays([self.X_train.columns,X_test.iloc[i]])
      post.append(l[np.argmax(np.multiply(self.likelihoods.loc[inn,:].prod(axis=0),self.priors))])

          
    return post

  def calculate_likelihoods(self):
    df = self.X_train.copy()
    df["label"] = self.y_train
    pp = pd.DataFrame([])
    columns = []
    for i in df.columns:
      for j in df[i].unique():
        columns.append((i,j))

    ind = pd.MultiIndex.from_tuples(columns)
    pp = pd.DataFrame(index = ind,columns = self.y_train.unique())
    for i in df.columns:
      for j in df[i].unique():
        for k in self.y_train.unique():
          pp.loc[(i,j),k] =  np.sum((df[i]==j)& (self.y_train==k))/np.sum(self.y_train==k)
       
      return pp
  
  def calculate_priors(self):
    priors = []
    for i in self.y_train.unique():
      priors.append(np.sum(self.y_train == i)/len(self.y_train))
    return priors 