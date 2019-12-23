def t_way_sparese_cov(neuron_value, number_of_neuron_layer, t_value):
    """
    1 把得到的神经元结点值写成激活/不激活
    2 神经元结点组合成t路
    3 对每个组合看是否被覆盖
    4 计算覆盖率
    :param neuron_value:
    :param number_of_neuron_layer:
    :param t_value:
    :return:
    """
    #神经元结点是否激活
    neuron_value_activate = neuron_value[:]
    bi_neuron_value_activate = [0 for i in range(neuron_value)]
    for i in range(neuron_value_activate):
        for j in range(number_of_neuron_layer)：
            if neuron_value_activate[i][j] > 0:
                bi_neuron_value_activate[i][j] = 1
            else:
                bi_neuron_value_activate[i][j] = 0

    #神经元结点值的组合
    neuron_com = []
    for i in range(number_of_neuron_layer):
        for j in range(i+1, number_of_neuron_layer):
            neuron_com.append([i, j])
    bi_neuron_value_activate