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
                #'conv2_sub1':tf.get_variable('conv2_sub1',[1,1,40,40],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #'conv2_sub2':tf.get_variable('conv2_sub2',[1,1,40,40],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #17*17*40->15*15*60->7*7*60
                'conv3':tf.get_variable('conv3',[3,3,40,60],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #'conv3_sub1':tf.get_variable('conv3_sub1',[1,1,60,60],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #'conv3_sub2':tf.get_variable('conv3_sub2',[1,1,60,60],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #7*7*60->120
                'conv4':tf.get_variable('conv4',[3,3,60,80],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv4_sub1':tf.get_variable('conv4_sub1',[1,1,80,40],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv4_sub2':tf.get_variable('conv4_sub2',[1,1,40,4],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'conv4_sub3':tf.get_variable('conv4_sub3',[1,1,20,4],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #'conv4_sub4':tf.get_variable('conv4_sub4',[1,1,20,4],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #'conv4_sub5':tf.get_variable('conv4_sub5',[1,1,20,4],initializer=tf.contrib.layers.xavier_initializer_conv2d())



                #fc1':tf.get_variable('fc1',[7*7*60,512],initializer=tf.contrib.layers.xavier_initializer()),
                #120->6
                #'fc2':tf.get_variable('fc2',[512,512],initializer=tf.contrib.layers.xavier_initializer()),
                #512->6
                #'fc3':tf.get_variable('fc3',[512,4],initializer=tf.contrib.layers.xavier_initializer()),'''
                }
        with tf.variable_scope("biases"):
            self.biases={
                'conv1':tf.get_variable('conv1',[20,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv2':tf.get_variable('conv2',[40,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2_sub1':tf.get_variable('conv2_sub1',[40,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2_sub2':tf.get_variable('conv2_sub2',[40,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv3':tf.get_variable('conv3',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3_sub1':tf.get_variable('conv3_sub1',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3_sub2':tf.get_variable('conv3_sub2',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv4':tf.get_variable('conv4',[80,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4_sub1':tf.get_variable('conv4_sub1',[40,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4_sub2':tf.get_variable('conv4_sub2',[4,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4_sub3':tf.get_variable('conv4_sub3',[4,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4_sub4':tf.get_variable('conv4_sub4',[4,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4_sub5':tf.get_variable('conv4_sub5',[4,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                #'fc1':tf.get_variable('fc1',[512,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                #'fc2':tf.get_variable('fc2',[512,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                #'fc3':tf.get_variable('fc3',[4,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }
        self.l1_loss=0
        for i in self.weights:
            self.l1_loss=self.l1_loss+tf.reduce_sum(tf.abs(self.weights[i]))
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
        #pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool1,_,_=tf.nn.fractional_max_pool(relu1,[1.0, 1.73, 1.73, 1.0])
        print "pool1:",pool1


        #第二层
        conv2=tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2=tf.nn.bias_add(conv2,self.biases['conv2'])
        #conv2 = batch_norm(conv2, 40, istrain,'conv2')
        relu2= tf.nn.relu(conv2)

        '''conv2_sub1=tf.nn.conv2d(relu2, self.weights['conv2_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_sub1=tf.nn.bias_add(conv2_sub1,self.biases['conv2_sub1'])
        relu2_sub1= tf.nn.relu(conv2_sub1)

        conv2_sub2=tf.nn.conv2d(relu2_sub1, self.weights['conv2_sub2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_sub2=tf.nn.bias_add(conv2_sub2,self.biases['conv2_sub2'])
        relu2_sub2= tf.nn.relu(conv2_sub2)



        #pool2=tf.nn.max_pool(relu2_sub2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')'''
        pool2,_,_=tf.nn.fractional_max_pool(relu2,[1.0, 1.73, 1.73, 1.0])
        #pool2=tf.nn.dropout(pool2,kepro)

        # 第三层
        conv3=tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID')
        conv3=tf.nn.bias_add(conv3,self.biases['conv3'])
        #conv3 = batch_norm(conv3, 60, istrain,'conv3')
        relu3= tf.nn.relu(conv3)

        '''conv3_sub1=tf.nn.conv2d(relu3, self.weights['conv3_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv3_sub1=tf.nn.bias_add(conv3_sub1,self.biases['conv3_sub1'])
        relu3_sub1= tf.nn.relu(conv3_sub1)

        conv3_sub2=tf.nn.conv2d(relu3_sub1, self.weights['conv3_sub2'], strides=[1, 1, 1, 1], padding='VALID')
        conv3_sub2=tf.nn.bias_add(conv3_sub2,self.biases['conv3_sub2'])
        relu3_sub2= tf.nn.relu(conv3_sub2)

        #pool3=tf.nn.max_pool(relu3_sub2, ksize= [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')'''
        pool3,_,_=tf.nn.fractional_max_pool(relu3,[1.0, 1.73, 1.73, 1.0])
        #pool3=tf.nn.dropout(pool3,kepro)

        #第四层
        conv4=tf.nn.conv2d(pool3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='VALID')
        #conv4 = batch_norm(conv4, 60, istrain,'conv4')
        conv4=tf.nn.bias_add(conv4,self.biases['conv4'])
        relu4= tf.nn.relu(conv4)

        pool4,_,_=tf.nn.fractional_max_pool(relu4,[1.0, 1.73, 1.73, 1.0])

        conv4_sub1=tf.nn.conv2d(pool4, self.weights['conv4_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv4_sub1=tf.nn.bias_add(conv4_sub1,self.biases['conv4_sub1'])
        relu4_sub1= tf.nn.relu(conv4_sub1)
        relu4_sub1,_,_=tf.nn.fractional_max_pool(relu4_sub1,[1.0, 1.41, 1.41, 1])

        conv4_sub2=tf.nn.conv2d(relu4_sub1, self.weights['conv4_sub2'], strides=[1, 1, 1, 1], padding='VALID')
        conv4_sub2=tf.nn.bias_add(conv4_sub2,self.biases['conv4_sub2'])
        relu4_sub2= tf.nn.relu(conv4_sub2)

        '''conv4_sub3=tf.nn.conv2d(relu4_sub2, self.weights['conv4_sub3'], strides=[1, 1, 1, 1], padding='VALID')
        conv4_sub3=tf.nn.bias_add(conv4_sub3,self.biases['conv4_sub3'])
        relu4_sub3= tf.nn.relu(conv4_sub3)'''

        '''conv4_sub4=tf.nn.conv2d(relu4_sub3, self.weights['conv4_sub4'], strides=[1, 1, 1, 1], padding='VALID')
        conv4_sub4=tf.nn.bias_add(conv4_sub4,self.biases['conv4_sub4'])
        relu4_sub4= tf.nn.relu(conv4_sub4)'''

        '''conv4_sub5=tf.nn.conv2d(relu4_sub4, self.weights['conv4_sub5'], strides=[1, 1, 1, 1], padding='VALID')
        conv4_sub5=tf.nn.bias_add(conv4_sub5,self.biases['conv4_sub5'])
        relu4_sub5= tf.nn.relu(conv4_sub5)'''
        #relu4_sub3=batch_norm(relu4_sub3,1000,istrain,'fc3')
        #print relu4_sub1
        print "relu4_sub1",conv4_sub2
        pool4=tf.nn.avg_pool(relu4_sub2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
        pool4=tf.squeeze(pool4)



        # 全连接层1，先把特征图转为向量
        '''flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        flatten=tf.nn.dropout(flatten,kepro)
        fc1=tf.matmul(flatten, self.weights['fc1'])
        fc1=fc1+self.biases['fc1']
        #fc1=batch_norm(fc1,1000,istrain,'fc1')
        fc_relu1=tf.nn.relu(fc1)

        drop1=tf.nn.dropout(fc_relu1,kepro)


        fc2=tf.matmul(drop1, self.weights['fc2'])
        fc2=fc2+self.biases['fc2']
        #fc2=batch_norm(fc2,1000,istrain,'fc2')
        fc_relu2=tf.nn.relu(fc2)


        fc3=tf.matmul(fc_relu2, self.weights['fc3'])
        fc3=fc3+self.biases['fc3']
        #fc3=batch_norm(fc3,1000,istrain,'fc3')

        return  fc3'''
        return  pool4


    #计算softmax交叉熵损失函数
    def sorfmax_loss(self,predicts,labels):

        #predicts=tf.nn.softmax(predicts)
        labels=tf.one_hot(labels,4)#self.weights['fc3'].get_shape().as_list()[1])
        loss=tf.nn.softmax_cross_entropy_with_logits(predicts,labels)
        loss =tf.reduce_mean(loss)# tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        self.cost= loss
        tf.scalar_summary("cost_function", loss)#损失函数值
        return self.cost
    #梯度下降
    def optimer(self,loss,lr=0.01,weight_decay=0.00005):
        '''l1=0
        for i in self.weights:
            grad=tf.gradients(loss,self.weights[i])[0]
            if grad is not None:
                l1+=tf.nn.l2_loss(tf.gradients(loss,self.weights[i])[0])'''
        #loss+=weight_decay*self.l1_loss
        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)



        return train_optimizer
    def accurate(self,test_inf,test_labels):
        correct_prediction = tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int32), test_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary("accuracy", accuracy)#损失函数值
        return accuracy


def train():

    shutil.rmtree('log')
    os.mkdir('log')



    tf.set_random_seed(123)

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

    logf=open('log.txt','w')

    with tf.Session() as session:

        summary_writer = tf.train.SummaryWriter('log', session.graph)



        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        '''if os.path.exists(os.path.join("model",'model_ori_race_regular.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model_ori_race_regular.ckpt'))'''
        #accuracy_np=session.run([accuracy],feed_dict = {batch_size: 400})




    #第一阶段：训练阶段
        max_iter=500000
        iter=0
        while iter<max_iter:
            #方案1
            summary_str,loss_np,_=session.run([merged_summary_op,loss,opti],feed_dict = {batch_size: 50})

            summary_writer.add_summary(summary_str, iter)
            if iter%400==0:
                print 'number iter:%s'%iter,'****************trainloss:%s'%loss_np
                str_test='train %s loss %s'%(iter,loss_np)
                logf.write(str_test+'\n')
            if iter%2000==0:
                accuracy_np=session.run([accuracy],feed_dict = {batch_size: 400})[0]
                print '***************test accruacy:',accuracy_np,'*******************'
                tf.train.Saver(max_to_keep=None).save(session, os.path.join('model','model_ori_race_regular.ckpt'))
                str_test='test %s accuracy %s'%(iter,accuracy_np)
                logf.write(str_test+'\n')

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

    logf.close()




train()







