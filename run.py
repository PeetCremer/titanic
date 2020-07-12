# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Iterable  

from data_conversion import load_data_dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

def plot_histogram(x, survived, bins, title=None, bin_labels=None):
  if bin_labels is not None:
    assert isinstance(bin_labels, Iterable)
    if isinstance(bins, Iterable):
      assert(len(bin_labels) == len(bins))
    else:
      assert(len(bin_labels) == bins)
  
  # Get numpy histogram to get statistics of x
  hist = np.histogram(x, bins=bins)
  print(hist)
  # Weights are the fraction of survivors for a passenger has survived times the count of passengers of that type
  weights = survived.astype(np.float64)
  lower_bound = hist[1][0]
  for i, upper_bound in enumerate(hist[1][1:]):
    weights[np.logical_and(x >= lower_bound, x <= upper_bound)] /= hist[0][i]
    lower_bound = upper_bound

  # print(survived)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  if title is not None:
    ax.set_title(title)
  ax.set_ylabel('survival ratio')
  _, bins, _ = ax.hist(x, bins=bins, weights=weights, facecolor='blue', edgecolor='gray')
  ax.set_xticks(bins)
  bin_centers = 0.5 * np.diff(bins) + bins[:-1]


  counts = hist[0]
  for count, x in zip(counts, bin_centers):  
    # # Label the raw counts
    # ax.annotate("{:.3f}".format(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
    #     xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages among the whole population
    percent = '%0.0f%%' % (100 * float(count) / np.sum(counts))
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')

  print(bin_centers)
  if bin_labels is not None:
    for bin_label, x in zip(bin_labels, bin_centers):
    # Label the bin labels
      ax.annotate(bin_label, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

  return fig, ax

def main():
  pkl_dir = './data'

  """
  # --- Direct data ---
  # Survivabilty by social class
  plot_histogram(train_dict["p_class"], train_dict["survived"], 3, title="Survivability by social class", bin_labels=["Upper", "Middle", "Lower"])
  # Survivability by gender -> Males just die. I expected some bias towards females here, but not that strong. 
  # plot_histogram(train_dict["sex"], train_dict["survived"], 2, title="Survivability by gender", bin_labels=["Male", "Female"])
  # Survivability by age -> Pretty uniform, except above 50 years where it drops significantly
  # plot_histogram(train_dict["age"], train_dict["survived"], [-1.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 90.0], title="Surivability by age")

  # Survivability by fare -> rises from 0.0 to 15.0, probably mostly constant up till 50.0 from there, then somewhat bigger after that. Probably strong correlation with social class
  # plot_histogram(train_dict["fare"], train_dict["survived"], [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], title="Survivability by fare")
  # Survivabilty by embarkation -> Some influence, surprisingly
  # plot_histogram(train_dict["embarked"], train_dict["survived"], [-1.0, 0.0, 0.5, 1.5, 2.5, 3.5], title="Survivability by point of embarkation")
  # Survivability by number of siblings and spouses -> 1 and 2 have highest chance of survival, from there chances drop fast  
  # plot_histogram(train_dict["sib_sp"], train_dict["survived"], [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], title="Survivability by number of siblings and spouses")
  # Survivability by parch -> 0 seems to be bad, beyond 3 seems to be really bad
  # plot_histogram(train_dict["parch"], train_dict["survived"], [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], title="Survivability by parch")
  # Survivability by cabin -> Having a cabin doubles chance of survival, but it does not seem to matter which type
  plot_histogram(train_dict["cabin"], train_dict["survived"], [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], title="Survivability by cabin")
  # --- Derived data ---
  # Survivability of males by age -> Ok for below 10 years, crappy beyond that
  male_mask = train_dict["sex"] == 1
  male_survived = train_dict["survived"][male_mask]
  male_age = train_dict["age"][male_mask]
  # plot_histogram(male_age, male_survived, [-1.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 90.0], title="Survivability of males by age")
  # Survivability of males by parch -> 0 is bad, 1 and 2 is ok, beyond that does not seem to exist
  male_parch = train_dict["parch"][male_mask]
  # plot_histogram(male_parch, male_survived, [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], title="Survivability of males by parch")
  # Survivability of females by age -> Pretty uniform across all ages
  female_mask = train_dict["sex"] == 2
  female_age = train_dict["age"][female_mask]
  female_survived = train_dict["survived"][female_mask]
  # plot_histogram(female_age, female_survived, [-1.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 90.0], title="Survivability of females by age")
  # Survivability of females by parch -> Uniform up to 3, beyond that seems to be really bad 
  female_parch = train_dict["parch"][female_mask]
  # plot_histogram(female_parch, female_survived, [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], title="Survivability of females by parch")

  # Plot
  plt.show()
  """

  # Create a train dataset by concatenating all the features that have been decided to be useful above
  def get_dataset_array(data_dict, keys):
    return np.column_stack([data_dict[key] for key in keys])

  # Get train dataset
  train_dict = load_data_dict(os.path.join(pkl_dir, "train.pkl"))
  X = get_dataset_array(train_dict, [
    "sex",
    "age",
    "p_class",
    "fare",   # TODO probably strong correlation with the above, use either one but not both / all three?
    # "cabin",   # TODO probably strong correlation with the above, use either one but not both / all three?
    # "embarked",
    # "sib_sp", 
    "parch"
  ])  
  y = train_dict["survived"]

  # Make a k-fold train / dev set-split
  # Use the same fraction of survivors in each set
  stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True)

  # Initialize classifier
  def get_classifier():
    return RandomForestClassifier(
      n_estimators=100, 
      criterion='gini', 
      max_depth=None, 
      min_samples_split=2, 
      min_samples_leaf=1, 
      min_weight_fraction_leaf=0.0, 
      max_features='auto', 
      max_leaf_nodes=None, 
      min_impurity_decrease=0.0, 
      min_impurity_split=None, 
      bootstrap=True, 
      oob_score=False, 
      n_jobs=None, 
      random_state=None, 
      verbose=0, 
      warm_start=False, 
      class_weight=None, 
      ccp_alpha=0.0, 
      max_samples=None
    )

  # Repeatedly fit the classifier to the splits
  # Average results of the splits
  num_eval_repetitions = 10
  scores = []
  for _ in range(num_eval_repetitions):
    for train_indices, test_indices in stratified_k_fold.split(X, y):
      X_train, y_train = X[train_indices], y[train_indices]
      X_test, y_test = X[test_indices], y[test_indices]

      # Get a fresh classifier (to prevent correlation between different splits)
      classifier = get_classifier()
      # Fit classifier to train data
      classifier.fit(X_train, y_train)

      # Evaluate score with test data
      score = classifier.score(X_test, y_test)
      scores.append(score)

  print("# SCORE: {} +- {}".format(np.mean(scores), np.std(scores)))


  # Fit classifier to training data

  """
  # Get submission dataset
  sub_dict = load_data_dict(os.path.join(pkl_dir, "test.pkl"))
  X_sub = get_dataset_array(sub_dict, [
    "sex",
    "age",
    "p_class",
    "fare",   # TODO probably strong correlation with the above, use either one but not both?
    "cabin",   # TODO probably strong correlation with the above, use either one but not both?
    "embarked",
    "sib_sp", 
    "parch"
  ])
  # Predict on the test dataset   
  y_sub = classifier.predict(X_sub)
  """


if __name__ == "__main__":
  main()


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session