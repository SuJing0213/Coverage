import numpy as np
import copy


def get_boundary(neuron_value, number_of_neuron):
    """
    对于当前所有样本的神经元输出值neuron_value， 从中找到每个神经元的最大值和最小值
    :param neuron_value: 神经元输出值 -1*number_of_value
    :param number_of_neuron: 神经元个数
    :return:
    """
    max_value = np.max(neuron_value, axis=0)
    min_value = np.min(neuron_value, axis=0)
    boundary = []
    for i in range(number_of_neuron):
        dic = {"max": max_value[i], "min": min_value[i]}
        boundary.append(dic)

    return boundary


def nbcov_and_snacov(neuron_value, number_of_neuron, boundary):
    """
    计算样本集的神经元边界覆盖率NBCov和强神经元激活覆盖率SNACov
    :param neuron_value:
    :param number_of_neuron:
    :param boundary:
    :return: NBCov, SNACov
    """
    count_upper = 0
    count_lower = 0
    for i in range(number_of_neuron):
        upper_flag, lower_flag = 0, 0
        for example in neuron_value:
            if example[i] > boundary[i]["max"] and upper_flag == 0:
                count_upper += 1
                upper_flag = 1
            elif example[i] < boundary[i]["min"] and lower_flag == 0:
                count_lower += 1
                lower_flag = 1
            if upper_flag == 1 and lower_flag == 1:
                break
    return (count_upper + count_lower)/(2*number_of_neuron), count_upper/number_of_neuron


def k_multisection_neuron_coverage(neuron_value, number_of_neuron, boundary, number_of_section):
    """
    得到k多节神经元覆盖率
    :param neuron_value: 对所有样本的神经元激活值 -1*number_of_neuron
    :param number_of_neuron: 神经元个数
    :param boundary: 激活值的边界
    :param number_of_section: 分成的节数量
    :return:
    """
    # 从训练集得到神经元出现过的最大最小值
    k_section_bound = []
    for i in range(number_of_neuron):
        temp = [boundary[i][min]]
        delta = (boundary[i][max] - boundary[i][min])/number_of_section
        k_bound = boundary[i][min]
        for _ in range(number_of_section):
            k_bound += delta
            temp.append(k_bound)
        k_section_bound.append(temp)

    count_k_section = [0 for i in range(number_of_section)]
    flag_k_section = [0 for i in range(number_of_section)]
    for i in range(number_of_neuron):
        for example in neuron_value:
            for k in range(number_of_section):
                if k_section_bound[k] <= example[i] <= k_section_bound[k+1] and flag_k_section[k] == 0:
                    flag_k_section[k] = 1
                    count_k_section[k] += 1
                    break
            if 0 not in flag_k_section:
                break
    return sum(count_k_section)/(number_of_section*number_of_neuron)


def top_k_neuron_cov(neuron_value, number_of_neuron_layer, k_value):
    """
    计算top-k覆盖率的值
    :param neuron_value: 神经元激活值 -1*number_of_neuron
    :param number_of_neuron_layer: 该层神经元个数
    :param k_value: 前k值
    :return:
    """
    neuron_value_copy = copy.deepcopy(neuron_value)
    top_k_neuron_index = [0 for i in range(10)]
    for i in range(len(neuron_value_copy)):
        for j in range(k_value):
            max_index = neuron_value_copy[i].index(max(neuron_value_copy[i]))
            neuron_value_copy[i][max_index] = -1
            if top_k_neuron_index[max_index] == 0:
                top_k_neuron_index[max_index] = 1
            elif top_k_neuron_index[max_index] == 1 and j == k_value - 1:
                break
    return sum(top_k_neuron_index)/number_of_neuron_layer


if __name__ == "__main__":

    layer_neuron_value = [[9, 1, 1, 1, 2, 1],
                          [1, 9, 1, 2, 1, 1],
                          [1, 2, 9, 1, 1, 1],
                          [2, 1, 1, 9, 1, 1]]
    cov_result = top_k_neuron_cov(layer_neuron_value, 6, 2)
    print(cov_result)
    print(layer_neuron_value)