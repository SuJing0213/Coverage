
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)


def classify_data_by_classes(
        data_set, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
    """
    将一个数据集按照类别，分别输出到不同的文件中
    """

    for i in range(len(data_set.images)):
        if data_set.labels[i] == 0:
            f = f0
        elif data_set.labels[i] == 1:
            f = f1
        elif data_set.labels[i] == 2:
            f = f2
        elif data_set.labels[i] == 3:
            f = f3
        elif data_set.labels[i] == 4:
            f = f4
        elif data_set.labels[i] == 5:
            f = f5
        elif data_set.labels[i] == 6:
            f = f6
        elif data_set.labels[i] == 7:
            f = f7
        elif data_set.labels[i] == 8:
            f = f8
        elif data_set.labels[i] == 9:
            f = f9
        image = np.array(data_set.images[i]).reshape(28, 28)
        for row in range(28):
            for col in range(28):
                print(image[row][col], end="    ", file=f)
        print("\n", file=f)


# 分类训练样本
f_train_0 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_0.txt", 'a')
f_train_1 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_1.txt", 'a')
f_train_2 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_2.txt", 'a')
f_train_3 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_3.txt", 'a')
f_train_4 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_4.txt", 'a')
f_train_5 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_5.txt", 'a')
f_train_6 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_6.txt", 'a')
f_train_7 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_7.txt", 'a')
f_train_8 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_8.txt", 'a')
f_train_9 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_9.txt", 'a')

classify_data_by_classes(mnist.train, f_train_0, f_train_1, f_train_2, f_train_3, f_train_4,
                         f_train_5, f_train_6, f_train_7, f_train_8, f_train_9)

# 分类测试样本
f_test_0 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_0.txt", 'a')
f_test_1 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_1.txt", 'a')
f_test_2 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_2.txt", 'a')
f_test_3 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_3.txt", 'a')
f_test_4 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_4.txt", 'a')
f_test_5 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_5.txt", 'a')
f_test_6 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_6.txt", 'a')
f_test_7 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_7.txt", 'a')
f_test_8 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_8.txt", 'a')
f_test_9 = open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\testing_data\test_images_class_9.txt", 'a')

classify_data_by_classes(mnist.test, f_test_0, f_test_1, f_test_2, f_test_3, f_test_4,
                         f_test_5, f_test_6, f_test_7, f_test_8, f_test_9)