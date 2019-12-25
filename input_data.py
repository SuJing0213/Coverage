import numpy as np


def load_neural_value(fpath, number_of_neuron):
    """
  根据所给文件把神经元输出值从文件中读取
  :param fpath:
  :param number_of_neuron:
  :return:
  """
    with open(fpath, 'r') as f:
        data = f.read()
        numlist = data.split()
        neural_value_ = []
        for number_str in numlist:
            number_float = float(number_str)
            neural_value_.append(number_float)
    neural_value_ = np.array(neural_value_).reshape([-1, number_of_neuron])

    return neural_value_
