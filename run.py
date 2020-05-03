# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from collections import Iterable  

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


def extract_data_from_data_frame(data_frame: pd.DataFrame):
  # The data contains: 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
  # 'PassengerId': ignore
  passenger_id = data_frame.get('PassengerId').to_numpy().astype(np.int64)
  # 'Survived': take as is (0 or 1) 
  survived = data_frame.get('Survived').to_numpy().astype(np.int64)
  # 'Pclass': take as is (1, 2, or 3)
  p_class = data_frame.get('Pclass').to_numpy().astype(np.int64)
  # 'Name': ignore
  name = data_frame.get('Name')
  # 'Sex' convert to 1 (male), 2 (female), and 0 (other / unknown)
  sex = data_frame.get('Sex')
  sex = sex.fillna('NONE')
  sex_list = []
  for s in sex.to_list():
    if s == "male":
      sex_list.append(1)
    elif s == "female":
      sex_list.append(2)
    elif s == "NONE":
      sex_list.append(0)
    else:
      print("Weird string for sex: {} still mapping to zero".format(s))
      sex_list.append(0)
  sex = np.array(sex_list, dtype=np.int64)
  # 'Age': take as is except where data is missing. For those rows, take -1
  age = data_frame.get('Age')
  age = age.fillna(-1)
  age = age.to_numpy()
  # 'SibSp': take as is
  sib_sp = data_frame.get('SibSp').to_numpy().astype(np.int64)
  # 'Parch': take as is
  parch = data_frame.get('Parch').to_numpy().astype(np.int64)
  # 'Ticket' number: take as is
  ticket_number = data_frame.get('Ticket').to_numpy()
  # 'Fare': take as is
  fare = data_frame.get('Fare').to_numpy().astype(np.float64)
  # 'Cabin': Convert to -1 where no data is available, 1 for A-type, 2 for B-type, 3 for C-type, 4 for D-type, 5 for E-type, 6 for F-type, 7 for G-type,
  # There is also a weird type "T", which we map to 0
  cabin = data_frame.get('Cabin')
  #print(cabin[0])
  cabin = cabin.fillna("NONE")
  cabin_list = []
  for c in cabin.to_list():
    if c.startswith('A'):
      cabin_list.append(1)
    elif c.startswith('B'):
      cabin_list.append(2)
    elif c.startswith('C'):
      cabin_list.append(3)
    elif c.startswith('D'):
      cabin_list.append(4)
    elif c.startswith('E'):
      cabin_list.append(5)
    elif c.startswith('F'):
      cabin_list.append(6)
    elif c.startswith('G'):
      cabin_list.append(7)
    elif c == "NONE":
      cabin_list.append(-1)
    else:
      cabin_list.append(0)
      print('Weird cabin string: {} Mapped to 0'.format(c))
    cabin = np.array(cabin_list, dtype=np.int64)
    # 'Embarked': Map C to 1, Q to 2, and S to 3. Map to -1 if embarkation port is not available and map other weird strings to zero
    embarked = data_frame.get('Embarked')
    embarked = embarked.fillna("NONE")
    embarked_list = []
    for e in embarked.to_list():
      if e == 'C': 
        embarked_list.append(1)
      elif e == 'Q':
        embarked_list.append(2)
      elif e == 'S':
        embarked_list.append(3)
      elif e == 'NONE':
        embarked_list.append(-1)
      else:
        # print('Unknown embarkation port: {}'.format(e))
        embarked_list.append(0)
        print('Weird embarkation string: {} Mapped to 0'.format(e))
    embarked = np.array(embarked_list, dtype=np.int64)

  return survived, p_class, sex, age, sib_sp, parch, ticket_number, fare, cabin, embarked

def plot_histogram(x, survived, bins, bin_labels=None):
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
  ax.set_ylabel('survival ratio')
  counts, bins, patches = ax.hist(x, bins=bins, weights=weights, facecolor='blue', edgecolor='gray')
  ax.set_xticks(bins)
  bin_centers = 0.5 * np.diff(bins) + bins[:-1]

  


  for count, x in zip(counts, bin_centers):  
    # # Label the raw counts
    # ax.annotate("{:.3f}".format(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
    #     xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages among the whole population
    percent = '%0.0f%%' % (100 * float(count) / counts.sum())
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
  data_dir = './kaggle/input'

  train_data_file = os.path.join(data_dir, 'train.csv')
  train_data_frame = pd.read_csv(train_data_file)

  test_data_file = os.path.join(data_dir, "test.csv")
  test_data_frame = pd.read_csv(test_data_file)

  # print(train_data_frame.columns)

  # numpy_stuff = train_data_frame.get('Sex').to_numpy()
  train_data_extracted = extract_data_from_data_frame(train_data_frame)
  survived, p_class, sex, age, sib_sp, parch, ticket_number, fare, cabin, embarked = train_data_extracted
  plot_histogram(p_class, survived, 3, bin_labels=["Upper", "Middle", "Lower"])
  
  plt.show()
  

  # test_data_extracted = extract_data_from_data_frame(test_data_frame)



if __name__ == "__main__":
  main()


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session