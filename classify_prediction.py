import tensorflow as tf
import numpy as np

# load_model_and_predict
for target in range(10):
    print("当前分类到第", target, "类\n")
    with open(r"C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\train_images_class_"+str(target)+".txt", 'r') as f:
        data = f.read()
        images = []
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            images.append(number_float)
    images = np.array(images).reshape([-1, 784])
    label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    sess = tf.Session()
    saver = tf.train.import_meta_graph(r'C:\Users\zz\Myproject\Mnist\Model\model.ckpt.meta')
    saver.restore(sess, r"C:\Users\zz\Myproject\Mnist\Model\model.ckpt")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    # keep_prob = graph.get_tensor_by_name("keep_prob:0")

    f_c = open(r'C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\class' +
               str(target)+'_correct_prediction.txt', 'w')
    f_w = open(r'C:\Users\zz\Myproject\Mnist\MNIST_data\training_data\class' +
               str(target)+'_wrong_prediction.txt', 'w')

    wrong_prediction = []
    wrong_picture = []
    correct_picture = []
    for i in range(len(images)):
        picture = np.array(images[i]).reshape(1, 28, 28, 1)
        feed_dict = {x: picture, y_: label}
        y = graph.get_tensor_by_name("layer7-fc3/output:0")
        yy = sess.run(y, feed_dict)
        classes = sess.run(tf.argmax(yy, 1))
        if classes != target:
            wrong_prediction.append(classes)
            wrong_picture.append(np.reshape(picture, [784]))
        else:
            correct_picture.append(np.reshape(picture, [784]))

    print(correct_picture, file=f_c)
    print(wrong_picture, file=f_w)
    print("第", target, "类的错误预测结果：", len(wrong_prediction), "\n", file=f_w)
    print(wrong_prediction, file=f_w)
    f_c.close()
    f_w.close()
