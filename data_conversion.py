
import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pickle

from typing import Dict

def extract_data_dict_from_data_frame(data_frame: pd.DataFrame, is_test=False):
  # The data contains: 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
  # 'PassengerId': ignore
  passenger_id = data_frame.get('PassengerId').to_numpy().astype(np.int64)

  # Survivded only exists if we are not in test mode
  if not is_test:
    # 'Survived': take as is (0 or 1) 
    # Some data seems to be broken in the test set, replace that with -1
    survived = data_frame.get('Survived')
    survived = survived.to_numpy().astype(np.int64)
  
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
  
  data_dict = {
    "p_class": p_class,
    "sex": sex,
    "age": age,
    "sib_sp": sib_sp,
    "parch": parch,
    "ticket_number": ticket_number,
    "fare": fare,
    "cabin": cabin,
    "embarked": embarked
  }
  if not is_test:
    data_dict["survived"] = survived

  return data_dict

def save_data_dict(datafile, data_dict: Dict):
  with open(datafile, 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data_dict(datafile):
  with open(datafile, 'rb') as handle:
      data_dict = pickle.load(handle)
  return data_dict


def convert_csv_to_pkl(input_csv, output_pkl, is_test=False):
    data_frame = pd.read_csv(input_csv)
    data_dict = extract_data_dict_from_data_frame(data_frame, is_test=is_test)
    save_data_dict(output_pkl, data_dict)


if __name__ == "__main__":
  csv_dir = './kaggle/input'
  pkl_dir = './data'

  os.makedirs(pkl_dir)
  convert_csv_to_pkl(os.path.join(csv_dir, 'train.csv'), os.path.join(pkl_dir, 'train.pkl'))
  convert_csv_to_pkl(os.path.join(csv_dir, 'test.csv'), os.path.join(pkl_dir, 'test.pkl'), is_test=True)
