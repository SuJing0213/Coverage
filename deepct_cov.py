
def two_way_sparse_cov(neuron_value, number_of_neuron_layer, boundary):
    """
    1 把得到的神经元结点值写成激活/不激活
    2 神经元结点组合成t路
    3 对每个组合看是否被覆盖
    4 计算覆盖率

    :param neuron_value:
    :param number_of_neuron_layer:
    :param boundary:
    :return:
    """
    # 神经元结点是否激活

    neuron_value_activate = neuron_value[:]
    bi_neuron_value_activate = [[0 for _ in range(number_of_neuron_layer)] for __ in range(len(neuron_value))]
    for i in range(len(neuron_value_activate)):
        for j in range(number_of_neuron_layer):
            if neuron_value_activate[i][j] > boundary:
                bi_neuron_value_activate[i][j] = 1
            else:
                bi_neuron_value_activate[i][j] = 0

    # 神经元结点值的组合
    count_combine_activate = 0
    count_combine = 0
    for i in range(number_of_neuron_layer):
        for j in range(i+1, number_of_neuron_layer):
            count_combine += 1
            flag_combine = [0, 0, 0, 0]
            for example in bi_neuron_value_activate:
                if example[i] == 0 and example[j] == 0:
                    flag_combine[0] = 1
                elif example[i] == 0 and example[j] == 1:
                    flag_combine[1] = 1
                elif example[i] == 1 and example[j] == 0:
                    flag_combine[2] = 1
                elif example[i] == 1 and example[j] == 1:
                    flag_combine[3] = 1
                if 0 not in flag_combine:
                    count_combine_activate += 1
                    break

    return count_combine_activate/count_combine
