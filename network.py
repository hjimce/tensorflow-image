#coding=utf-8
import  tensorflow as tf
from  data_encoder_decoeder import  encode_to_tfrecords,decode_from_tfrecords,get_batch,get_test_batch
import  cv2
import  os
from  layers import  batch_norm
#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
class network(object):
    def __init__(self):
        with tf.variable_scope("weights"):
            self.weights={
                #39*39*3->36*36*20->18*18*20
                'conv1':tf.get_variable('conv1',[4,4,3,20],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #18*18*20->16*16*40->8*8*40
                'conv2':tf.get_variable('conv2',[3,3,20,40],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #8*8*40->6*6*60->3*3*60
                'conv3':tf.get_variable('conv3',[3,3,40,60],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                #3*3*60->120
                'fc1':tf.get_variable('fc1',[3*3*60,120],initializer=tf.contrib.layers.xavier_initializer()),
                #120->6
                'fc2':tf.get_variable('fc2',[120,6],initializer=tf.contrib.layers.xavier_initializer()),
                }
        with tf.variable_scope("biases"):
            self.biases={
                'conv1':tf.get_variable('conv1',[20,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2':tf.get_variable('conv2',[40,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3':tf.get_variable('conv3',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1':tf.get_variable('fc1',[120,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2':tf.get_variable('fc2',[6,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }

    def inference(self,images):
        # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 39,39, 3])# [batch, in_height, in_width, in_channels]
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理



        #第一层
        conv1=tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID')
        #conv1=tf.nn.bias_add(conv1,self.biases['conv1'])

        conv1 = batch_norm(conv1, 20, True,'conv1')
        relu1= tf.nn.relu(conv1)
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        #第二层
        conv2=tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID')
        #conv2=tf.nn.bias_add(conv2,self.biases['conv2'])
        conv2 = batch_norm(conv2, 40, True,'conv2')
        relu2= tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 第三层
        conv3=tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID')
        #conv3=tf.nn.bias_add(conv3,self.biases['conv3'])
        conv3 = batch_norm(conv3, 60, True,'conv3')
        relu3= tf.nn.relu(conv3)
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        #drop1=tf.nn.dropout(flatten,0.5)
        fc1=tf.matmul(flatten, self.weights['fc1'])
        #fc1=fc1+self.biases['fc1']
        fc1=batch_norm(fc1,1000,True,'fc1')

        fc_relu1=tf.nn.relu(fc1)
        fc2=tf.matmul(fc_relu1, self.weights['fc2'])
        #fc2=fc2+self.biases['fc2']
        fc2=batch_norm(fc2,1000,True,'fc2')

        return  fc2
    def inference_test(self,images):
          # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 39,39, 3])# [batch, in_height, in_width, in_channels]
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理



        #第一层
        conv1=tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID')
        #conv1=tf.nn.bias_add(conv1,self.biases['conv1'])
        conv1 = batch_norm(conv1, 20,False,'conv1')
        relu1= tf.nn.relu(conv1)
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        #第二层
        conv2=tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID')
        #conv2=tf.nn.bias_add(conv2,self.biases['conv2'])
        conv2 = batch_norm(conv2, 40, False,'conv2')
        relu2= tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 第三层
        conv3=tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID')
        #conv3=tf.nn.bias_add(conv3,self.biases['conv3'])
        conv3 = batch_norm(conv3, 60,False,'conv3')
        relu3= tf.nn.relu(conv3)
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        #drop1=tf.nn.dropout(flatten,0.5)
        fc1=tf.matmul(flatten, self.weights['fc1'])
        #fc1=fc1+self.biases['fc1']
        fc1=batch_norm(fc1,1000,False,'fc1')

        fc_relu1=tf.nn.relu(fc1)
        fc2=tf.matmul(fc_relu1, self.weights['fc2'])
        #fc2=fc2+self.biases['fc2']
        fc2=batch_norm(fc2,1000,False,'fc2')

        return fc2

    #计算softmax交叉熵损失函数
    def sorfmax_loss(self,predicts,labels):
        predicts=tf.nn.softmax(predicts)
        labels=tf.one_hot(labels,self.weights['fc2'].get_shape().as_list()[1])
        loss =-tf.reduce_mean(labels * tf.log(predicts))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        self.cost= loss
        return self.cost
    #梯度下降
    def optimer(self,loss,lr=0.1):
        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return train_optimizer


def train():
    #encode_to_tfrecords("data/train.txt","data",'train.tfrecords',(45,45))
    image,label=decode_from_tfrecords('data/train.tfrecords')
    batch_image,batch_label=get_batch(image,label,batch_size=50,crop_size=39)#batch 生成测试







   #网络链接,训练所用
    net=network()
    inf=net.inference(batch_image)
    loss=net.sorfmax_loss(inf,batch_label)
    opti=net.optimer(loss)


    #验证集所用
    #encode_to_tfrecords("data/val.txt","data",'val.tfrecords',(45,45))
    test_image,test_label=decode_from_tfrecords('data/val.tfrecords',num_epoch=None)
    test_images,test_labels=get_test_batch(test_image,test_label,batch_size=120,crop_size=39)#batch 生成测试
    test_inf=net.inference_test(test_images)
    correct_prediction = tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int32), test_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





    init=tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        max_iter=100000
        iter=0
        '''if os.path.exists(os.path.join("model",'model.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join("model",'model.ckpt'))'''
        while iter<max_iter:
            loss_np,_,label_np,image_np,inf_np=session.run([loss,opti,batch_label,batch_image,inf])
            #print image_np.shape
            #cv2.imshow(str(label_np[0]),image_np[0])
            #print label_np[0]
            #cv2.waitKey()
            #print label_np
            if iter%50==0:
                print 'trainloss:',loss_np
            if iter%500==0:
                accuracy_np=session.run([accuracy])
                print '***************test accruacy:',accuracy_np,'*******************'
                tf.train.Saver(max_to_keep=None).save(session, os.path.join('model','model.ckpt'))
            iter+=1





        coord.request_stop()#queue需要关闭，否则报错
        coord.join(threads)

train()




