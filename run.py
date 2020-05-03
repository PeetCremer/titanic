# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import matplotlib.pyplot as plt
from collections import Iterable  


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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
  data_input_dir = './kaggle/input'
  data_conversion_dir = 


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