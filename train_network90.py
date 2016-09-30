#coding=utf-8

import  tensorflow as tf
from  data_encoder_decoeder import  encode_to_tfrecords,decode_from_tfrecords,get_batch,get_test_batch
import  cv2
import  os
from  layers import  batch_norm
from  compress import prune,binary,quantization
import shutil
#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
class network(object):
    def __init__(self):
        with tf.variable_scope("weights"):
            self.weights={
                #76*76*3->74*74*20->37*37*20
                'conv1':tf.get_variable('conv1',[3,3,3,20],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #37*37*20->35*35*40->17*17*40
                'conv2':tf.get_variable('conv2',[3,3,20,40],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #17*17*40->15*15*60->7*7*60
                'conv3':tf.get_variable('conv3',[3,3,40,60],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #7*7*60->120
                'fc1':tf.get_variable('fc1',[7*7*60,512],initializer=tf.contrib.layers.xavier_initializer()),
                #120->6
                'fc2':tf.get_variable('fc2',[512,512],initializer=tf.contrib.layers.xavier_initializer()),
                #512->6
                'fc3':tf.get_variable('fc3',[512,4],initializer=tf.contrib.layers.xavier_initializer()),
                }
        with tf.variable_scope("biases"):
            self.biases={
                'conv1':tf.get_variable('conv1',[20,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2':tf.get_variable('conv2',[40,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3':tf.get_variable('conv3',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1':tf.get_variable('fc1',[512,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2':tf.get_variable('fc2',[512,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc3':tf.get_variable('fc3',[4,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }

    def inference(self,images,istrain=True):

        if istrain:
            kepro=0.5
        else:
            kepro=1


        # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 76,76, 3])# [batch, in_height, in_width, in_channels]
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理



        #第一层
        conv1=tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        #conv1 = batch_norm(conv1, 20, istrain,'conv1')
        relu1= tf.nn.relu(conv1)
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        #第二层
        conv2=tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2=tf.nn.bias_add(conv2,self.biases['conv2'])
        #conv2 = batch_norm(conv2, 40, istrain,'conv2')
        relu2= tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 第三层
        conv3=tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID')
        conv3=tf.nn.bias_add(conv3,self.biases['conv3'])
        #conv3 = batch_norm(conv3, 60, istrain,'conv3')
        relu3= tf.nn.relu(conv3)
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        flatten=tf.nn.dropout(flatten,kepro)
        fc1=tf.matmul(flatten, self.weights['fc1'])
        fc1=fc1+self.biases['fc1']
        #fc1=batch_norm(fc1,1000,istrain,'fc1')
        fc_relu1=tf.nn.relu(fc1)



        fc2=tf.matmul(fc_relu1, self.weights['fc2'])
        fc2=fc2+self.biases['fc2']
        #fc2=batch_norm(fc2,1000,istrain,'fc2')
        fc_relu2=tf.nn.relu(fc2)


        fc3=tf.matmul(fc_relu2, self.weights['fc3'])
        fc3=fc3+self.biases['fc3']
        #fc3=batch_norm(fc3,1000,istrain,'fc3')

        return  fc3


    #计算softmax交叉熵损失函数
    def sorfmax_loss(self,predicts,labels):
        predicts=tf.nn.softmax(predicts)
        labels=tf.one_hot(labels,self.weights['fc3'].get_shape().as_list()[1])
        loss =-tf.reduce_mean(labels * tf.log(predicts))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        self.cost= loss
        return self.cost
    #梯度下降
    def optimer(self,loss,lr=0.1,weight_decay=0.001):
        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)



        return train_optimizer
    def accurate(self,test_inf,test_labels):
        correct_prediction = tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int32), test_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.scalar_summary("accuracy", accuracy)#损失函数值
        return accuracy


def train():

    shutil.rmtree('log')
    os.mkdir('log')



    tf.set_random_seed(1)

    batch_size=tf.placeholder(tf.int32,[],'batch_size')
    #encode_to_tfrecords("data/race/train.txt","data/race",'train.tfrecords',(90,90))
    image,label=decode_from_tfrecords('data/race/train.tfrecords')
    batch_image,batch_label=get_batch(image,label,batch_size=batch_size,crop_size=76)#batch 生成测试







   #网络链接,训练所用
    net=network()
    inf=net.inference(batch_image,True)
    loss=net.sorfmax_loss(inf,batch_label)
    opti=net.optimer(loss)

    #tf.histogram_summary("weights",tf.all_variables()[0])#卷积层conv1滤波器w1
    tf.scalar_summary("cost_function", loss)#损失函数值




    #验证集所用
    #encode_to_tfrecords("data/race/val.txt","data/race",'val.tfrecords',(90,90))
    test_image,test_label=decode_from_tfrecords('data/race/val.tfrecords',num_epoch=None)
    test_images,test_labels=get_test_batch(test_image,test_label,batch_size=batch_size,crop_size=76)#batch 生成测试
    test_inf=net.inference(test_images,False)
    accuracy=net.accurate(test_inf,test_labels)

    tf.add_to_collection('test_images',test_images)
    tf.add_to_collection('test_inf',test_inf)
    tf.add_to_collection('batch_size',batch_size)



    #修剪网络
    prune_obj=prune()
    prune_weights=prune_obj.prune_network()
    prune_opti=prune_obj.optimer(loss)
    [zero_num,all_num]=prune_obj.cout_zeros()


    #二值网络
    '''binary_obj=binary()
    binary_weights=binary_obj.binary_network()
    binary_opti=binary_obj.optimer(loss)
    [zero_num,all_num]=binary_obj.cout_ones()'''





    #quantion=quantization()

    init=tf.initialize_all_variables()
    merged_summary_op = tf.merge_all_summaries()

    with tf.Session() as session:

        summary_writer = tf.train.SummaryWriter('log', session.graph)



        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if os.path.exists(os.path.join("model",'model_ori_race.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model_ori_race.ckpt'))
        accuracy_np=session.run([accuracy],feed_dict = {batch_size: 400})
        print "ori accuracy:",accuracy_np

    #第一阶段：训练阶段
        max_iter=100000
        iter=0
        while iter<max_iter:
            #方案1
            summary_str,loss_np,_=session.run([merged_summary_op,loss,opti],feed_dict = {batch_size: 64})

            summary_writer.add_summary(summary_str, iter)
            if iter%50==0:
                print 'number iter:%s'%iter,'****************trainloss:%s'%loss_np
            if iter%500==0 and iter!=0:
                accuracy_np=session.run([accuracy],feed_dict = {batch_size: 400})
                print '***************test accruacy:',accuracy_np,'*******************'
                tf.train.Saver(max_to_keep=None).save(session, os.path.join('model','model_ori_race.ckpt'))

            iter+=1






    #第二阶段：prune阶段
        '''if os.path.exists(os.path.join("model",'model_ori.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model_ori.ckpt'))
        zero_num_np,all_num_np=session.run([zero_num,all_num])
        print "原始还未修剪，参数为0个数：",zero_num_np,"***,参数总数：",all_num_np
        for ptime in range(10):
            session.run(prune_weights)
            zero_num_np,all_num_np=session.run([zero_num,all_num])
            print "修剪后，初次参数为0个数：",zero_num_np,"***,参数总数：",all_num_np
            iter=0
            max_iter=10000
            while iter<max_iter:
                if iter%500==0 and iter!=0:
                    zero_num_np,all_num_np=session.run([zero_num,all_num])
                    print "修剪后训练过程中，参数为0个数：",zero_num_np,"***,参数总数：",all_num_np
                    accuracy_np=session.run([accuracy])
                    print '***************test accruacy:',accuracy_np,'*******************'
                    tf.train.Saver(max_to_keep=None).save(session, os.path.join('model','model_prune_finetuning.ckpt'))

                loss_np,_=session.run([loss,prune_opti])
                if iter%100==0:
                    print '迭代第',iter,'次trainloss:',loss_np

                iter+=1'''










        coord.request_stop()#queue需要关闭，否则报错
        coord.join(threads)






train()







