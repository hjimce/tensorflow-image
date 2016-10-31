#coding=utf-8

import  tensorflow as tf
from  preprocess.data_encoder_decoeder import  encode_to_tfrecords,decode_from_tfrecords,get_batch,get_test_batch,preload_data,batch_inputs
import  cv2
import  os
from  algorithms.layers import  batch_norm
from  algorithms.compress import prune,binary,quantization
import  time
import shutil
import  sys
sys.path.append("../")
#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
class network(object):
    def __init__(self):

        with tf.variable_scope("weights"):
            self.weights={
                'conv1':tf.get_variable('conv1',[3,3,3,20],initializer=tf.truncated_normal_initializer(0,0.1)),
                #'conv1_sub1':tf.get_variable('conv1_sub1',[1,1,2,100],initializer=tf.truncated_normal_initializer(0,0.1)),


                'conv2':tf.get_variable('conv2',[3,3,20,40],initializer=tf.truncated_normal_initializer(0,0.1)),
                'conv3':tf.get_variable('conv3',[3,3,40,60],initializer=tf.truncated_normal_initializer(0,0.1)),
                #3*3*60->120
                'fc1':tf.get_variable('fc1',[3*3*60,120],initializer=tf.truncated_normal_initializer(0,0.1)),
                #120->6
                'fc2':tf.get_variable('fc2',[120,6],initializer=tf.truncated_normal_initializer(0,0.1)),
                }
        with tf.variable_scope("biases"):
            self.biases={
                'conv1':tf.get_variable('conv1',[20,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv2':tf.get_variable('conv2',[40,],initializer=tf.constant_initializer(value=0., dtype=tf.float32)),

                'conv3':tf.get_variable('conv3',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'fc1':tf.get_variable('fc1',[120,],initializer=tf.constant_initializer(value=0., dtype=tf.float32)),
                'fc2':tf.get_variable('fc2',[6,],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
            }

        self.l2_loss=0.
        for i in self.weights:
            print self.weights[i].name
            self.l2_loss=self.l2_loss+tf.reduce_sum(tf.mul(self.weights[i],self.weights[i]))
    def inference(self,images,istrain=True):

        if istrain:
            kepro=0.5
        else:
            kepro=1

        # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 39,39, 3])# [batch, in_height, in_width, in_channels]
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理



        #第一层
        conv1=tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1= tf.nn.relu(conv1)

        '''conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1=tf.nn.conv2d(conv1, self.weights['conv1_sub1'], strides=[1, 1, 1, 1], padding='VALID')
        conv1=tf.nn.bias_add(conv1,self.biases['conv1'])'''












        pool1=tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')



        #第二层
        conv2=tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2=tf.nn.bias_add(conv2,self.biases['conv2'])
        relu2= tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 第三层
        conv3=tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID')
        conv3=tf.nn.bias_add(conv3,self.biases['conv3'])
        relu3= tf.nn.relu(conv3)
        pool3=tf.nn.max_pool(relu3, ksize= [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        drop3=tf.nn.dropout(pool3,kepro)


        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(drop3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        flatten=tf.nn.dropout(flatten,kepro)
        fc1=tf.matmul(flatten, self.weights['fc1'])+self.biases['fc1']
        relufc1=tf.nn.relu(fc1)
        dropfc1=tf.nn.dropout(relufc1,kepro)




        fc2=tf.matmul(dropfc1, self.weights['fc2'])+self.biases['fc2']

        return  fc2



    #计算softmax交叉熵损失函数
    def sorfmax_loss(self,predicts,labels):

        labels=tf.one_hot(labels,self.weights['fc2'].get_shape().as_list()[1])
        loss=tf.nn.softmax_cross_entropy_with_logits(predicts,labels)
        loss =tf.reduce_mean(loss)# tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        self.cost= loss
        tf.scalar_summary("cost_function", loss)#损失函数值
        return self.cost
    #梯度下降
    def optimer(self,loss,lr=0.001,weight_decay=0.0005):
        '''l1=0
        for i in self.weights:
            grad=tf.gradients(loss,self.weights[i])[0]
            if grad is not None:
                l1+=tf.nn.l2_loss(tf.gradients(loss,self.weights[i])[0])'''
        loss+=weight_decay*self.l2_loss
        train_optimizer = tf.train.MomentumOptimizer(lr,0.9).minimize(loss)



        return train_optimizer
    def accurate(self,test_inf,test_labels):
        correct_prediction = tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int32), test_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary("accuracy", accuracy)#损失函数值
        return accuracy
def create_data(root="../data/mutil-light",labelfile='train.txt"',outfile='train.tfrecords',resize=45):
    encode_to_tfrecords(os.path.join(root,labelfile),root,outfile,(resize,resize))

def train(ori_size=45,crop_size=39,train_batchsize=64,test_batchsize=240):
    shutil.rmtree('log')
    os.mkdir('log')
    with tf.device('/cpu:0'):
        tf.set_random_seed(123)
        batch_size=tf.placeholder(tf.int32,[],'batch_size')

        image,label=decode_from_tfrecords('../data/mutil-light/train.tfrecords')
        batch_image,batch_label=get_batch(image,label,batch_size=batch_size,crop_size=crop_size)#batch 生成测试
            #验证集所用
        #encode_to_tfrecords("../data/mutil-light/val.txt","../data/mutil-light",'val.tfrecords',(45,45))
        test_image,test_label=decode_from_tfrecords('../data/mutil-light/val.tfrecords',num_epoch=None)
        test_images,test_labels=get_test_batch(test_image,test_label,batch_size=batch_size,crop_size=crop_size,ori_size=ori_size)#batch 生成测试

    #batch_image, batch_label = batch_inputs('../data/mutil-light/train.tfrecords',64)


        #网络链接,训练所用
    net=network()
    inf=net.inference(batch_image,True)
    loss=net.sorfmax_loss(inf,batch_label)
    opti=net.optimer(loss)

    #tf.histogram_summary("weights",tf.all_variables()[0])#卷积层conv1滤波器w1






    test_inf=net.inference(test_images,False)
    accuracy=net.accurate(test_inf,test_labels)



    print "faf"





    '''tf.add_to_collection('test_images',test_images)
    tf.add_to_collection('test_inf',test_inf)
    tf.add_to_collection('batch_size',batch_size)'''



    #修剪网络
    '''prune_obj=prune()
    prune_weights=prune_obj.prune_network()
    prune_opti=prune_obj.optimer(loss)
    [zero_num,all_num]=prune_obj.cout_zeros()'''


    #二值网络
    '''binary_obj=binary()
    binary_weights=binary_obj.binary_network()
    binary_opti=binary_obj.optimer(loss)
    [zero_num,all_num]=binary_obj.cout_ones()'''





    #quantion=quantization()

    init=tf.initialize_all_variables()
    merged_summary_op = tf.merge_all_summaries()

    logf=open('log.txt','w')
    session=tf.Session()

    summary_writer = tf.train.SummaryWriter('log', session.graph)
    session.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session,coord=coord)

    '''if os.path.exists(os.path.join("model",'model_ori_race_regular.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model_ori_race_regular.ckpt'))'''
    try:
        t0=time.clock()
        for iter in xrange(500000):
            if coord.should_stop():
                break
            if iter%2000==0:
                accuracy_np,test_inf_np=session.run([accuracy,test_inf],feed_dict = {batch_size: test_batchsize})
                print '***************test accruacy:',accuracy_np,'*******************'
                tf.train.Saver(max_to_keep=None).save(session, os.path.join('model','model_%s.ckpt'%str(iter)))
                str_test='test %s accuracy %s'%(iter,accuracy_np)
                logf.write(str_test+'\n')
                #训练
            summary_str,loss_np,_=session.run([merged_summary_op,loss,opti],feed_dict = {batch_size: train_batchsize})
            summary_writer.add_summary(summary_str, iter)
            if iter%500==0:
                print (time.clock()-t0)*1000
                t0=time.clock()
                print 'number iter:%s'%iter,'****************trainloss:%s'%loss_np
                str_test='train %s loss %s'%(iter,loss_np)
                logf.write(str_test+'\n')
        logf.close()
    except Exception, e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)











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












train()







