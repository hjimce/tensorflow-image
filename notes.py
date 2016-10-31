#coding=utf-8
#tenosorflow 记录
1、tensorflow在使用的时候，训练数据部分代码一定要定义再cpu上，不然会默认占用GPU，反而导致速度下降，比如下面：
    with tf.device('/cpu:0'):
        batch_image, batch_label = batch_inputs('../data/mutil-light/train.tfrecords',64)#训练数据定义再cpu上
        test_images,test_labels=get_test_batch(test_image,test_label,batch_size=batch_size,crop_size=crop_size)#测试数据也定义再cpu上
    我在训练的时候，发现如果没有加上：with tf.device('/cpu:0')来强制定义数据分配再cpu上,这句代码每批训练时间是：137s左右，然而如果加入这句代码，训练时间差不多是32s。
    据此我们可以明白，使用tensorflow的时候，需要人为定义device是必要的。不管是训练数据还是测试数据，尽量不要占用gpu显存空间，速度影响相当巨大呀.
    网络模型的参数、计算过程建议定义在gpu上，也就是默认设置。
2、在调用函数tf.train.batch等函数的时候，需要设置多线程，因为tensorflow默认的线程数是1，经过实验：不管是验证集数据、还是训练集数据，都需要设置。之前自己只设置了训练集为8线程
    速度是32s左右，然后把验证集也设置成了8线程，结果速度提升至26s