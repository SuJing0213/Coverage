import numpy as np
import tensorflow as tf

np.set_printoptions(precision=8, suppress=True, threshold=np.nan)


def get_neural_value(source_path, result_path, tensor_name, number_of_value):
    """
    给出图片数据集，输出每张图片经过网络后，每个位置上的神经元输出值
    """
    with open(source_path, 'r') as source_file:
        data = source_file.read()
        image = []
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            image.append(number_float)
    images = np.array(image).reshape([-1, 784])
    label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    sess = tf.Session()
    saver = tf.train.import_meta_graph(r'C:\Users\zz\Myproject\Mnist\Model\model.ckpt.meta')
    saver.restore(sess, r"C:\Users\zz\Myproject\Mnist\Model\model.ckpt")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")

    with open(result_path, 'w') as result_file:
        for k in range(len(images)):
            picture = np.array(images[k]).reshape(1, 28, 28, 1)
            feed_dict = {x: picture, y_: label}
            tensor = graph.get_tensor_by_name(tensor_name)
            layer_output = sess.run(tensor, feed_dict)
            layer_output = np.array(layer_output).reshape([number_of_value])
            for value in layer_output:
                print(value, end='    ', file=result_file)
            print("\n", file=result_file)


# ------------------------------------------训练集神经元输出----------------------------------------------------
# 计算预测正确训练集图片的神经元输出
for i in range(10):
    Source_path = r'C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\class' + str(i) + \
                  '_correct_prediction.txt'

    # # 第一个池化层的输出
    # Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\training_data\correct_neural_value\conv_1\class' + \
    #               str(i) + '_correct_NeuralValue.txt'
    # get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="conv1:0", number_of_value=14*14*32
    #
    # # 第二个池化层的输出
    # Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\training_data\correct_neural_value\conv_2\class' + \
    #               str(i) + '_correct_NeuralValue.txt'
    # get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="conv2:0", number_of_value=7*7*64)

    # 第五层全连接层输出
    Result_path = r'C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\correct_neural_value\fc1\class' + \
                  str(i) + '_correct_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="layer5-fc1/fc1:0", number_of_value=256)

    # 第六层全连接层输出
    Result_path = r'C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\correct_neural_value\fc2\class' + \
                  str(i) + '_correct_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="layer6-fc2/fc2:0", number_of_value=128)



# #计算预测错误训练集图片的神经元输出
# # for i in range(10):
# #     Source_path = r'C:\Users\Myproject\LeNet5\MNIST_data\training_data\class' + str(i) + \
# #                   '_wrong_prediction.txt'
# #     Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\training_data\wrong_neural_value\class' + \
# #                   str(i) + '_wrong_NeuralValue.txt'
# #     get_neural_value(source_path=Source_path, result_path=Result_path)

# # ------------------------------------------测试集神经元输出----------------------------------------------------
# #计算预测正确训练集图片的神经元输出
# for i in range(10):
#     Source_path = r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\class' + str(i) + \
#                   '_correct_prediction.txt'
#
#     # # 第一个卷积层的输出
#     # Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\correct_neural_value\conv_1\class' + \
#     #               str(i) + '_correct_NeuralValue.txt'
#     # get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="conv_1:0", number_of_value=28*28*32)
#     #
#     # #第二个卷积层的输出
#     # Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\correct_neural_value\conv_2\class' + \
#     #               str(i) + '_correct_NeuralValue.txt'
#     # get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="conv_2:0", number_of_value=14*14*64)
#
#     #第三个全连接层输出
#     Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\correct_neural_value\fc_3\class' + \
#                   str(i) + '_correct_NeuralValue.txt'
#     get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="fc_3:0", number_of_value=1024)

# 计算预测错误训练集图片的神经元输出
# for i in range(1,10):
#     Source_path = r'C:\Users\Myproject\LeNet5\MNIST_data\adversarial_data\wrong_prediction_to_[' + str(i) + \
#                   ']_label.txt'

    # # 第一个卷积层的输出
    # Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\correct_neural_value\conv_1\class' + \
    #               str(i) + '_correct_NeuralValue.txt'
    # get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="conv_1:0", number_of_value=28*28*32)
    #
    # # 第二个卷积层的输出
    # Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\correct_neural_value\conv_2\class' + \
    #               str(i) + '_correct_NeuralValue.txt'
    # get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="conv_2:0", number_of_value=14*14*64)

    # # 第三个全连接层输出
    # Result_path = r'C:\Users\Myproject\LeNet5\MNIST_data\adversarial_data\wrong_prediction_to_[' + \
    #               str(i) + ']_NeuralValue.txt'
    # get_neural_value(source_path=Source_path, result_path=Result_path, tensor_name="fc_3:0", number_of_value=1024)
