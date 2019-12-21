# neuron_coverage in DeepXplore
def neuron_coverage(neuron_value, neuron_num, activate_bound):
    """
    以一定的激活阈值，计算样本集合的神经元激活覆盖率
    :param neuron_value: 所有样本的神经元输出值 -1*number_of_neural
    :param neuron_num: 网络中的神经元个数
    :param activate_bound: 激活阈值
    :return:
    """
    activated_num = 0.0
    for i in range(neuron_num):
        for example in neuron_value:
            if example[i] > activate_bound:
                activated_num += 1
                break
    coverage = activated_num/neuron_num
    return coverage


