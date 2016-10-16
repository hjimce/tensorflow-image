#coding=utf-8
import  tensorflow as tf
class prune(object):
    def __init__(self):
        self.optmasks=[]

#修剪网络,也就是把绝对值小于阈值的参数，全部置为0
    def prune_network(self,threshold=0.01):
        newvs=[]
        self.optmasks=[]
        for v in tf.all_variables():
        #只对权重处理，如果是偏置项不做处理，偏置项是一维的矩阵
            mask=tf.cond(tf.rank(v)>1,lambda:tf.to_float(tf.greater_equal(tf.abs(v), tf.ones_like(v)*threshold)),
                       lambda:tf.ones_like(v))
            #mask=tf.to_float(tf.greater_equal(tf.abs(v), tf.ones_like(v)*threshold))

            prunev=tf.mul(mask,v)
            newv=v.assign(prunev)
            newvs.append(newv)

            self.optmasks.append(mask)



        return  newvs


    def cout_zeros(self):
        zeros_num=0.
        all_num=0.
        for v in tf.all_variables():
            zeros_num += tf.reduce_sum(tf.to_float(tf.less(tf.abs(v), tf.ones_like(v)*0.0001)))#统计0的个数
            all_num += tf.reduce_sum(tf.ones_like(v))
        return [zeros_num,all_num]
        #梯度下降
    def optimer(self,loss,lr=0.1,weight_decay=0.001):
        grads=tf.gradients(loss, tf.all_variables())
        train_optimizers=[]
        for mask,grad,v in zip(self.optmasks,grads,tf.all_variables()):
            if grad is not None:
                lrate=tf.mul(mask,tf.ones_like(v)*lr)#在mask中，0就是被修剪去的参数
                train_optimizer =v.assign(v-tf.mul(grad,lrate))
                train_optimizers.append(train_optimizer)


        return train_optimizers


#二值网络
class binary(object):
    def __init__(self):
        self.optmasks=[]

#二值网络，把参数数值大于0的置为1，数值小于0的置为-1
    def binary_network(self,threshold=0.):
        newvs=[]
        for v in tf.all_variables():
            mask_positive=tf.to_float(tf.greater_equal(v, tf.ones_like(v)*threshold))
            mask_negative=tf.to_float(tf.less(v, tf.ones_like(v)*threshold))
            prunev=tf.cond(tf.rank(v)>1,lambda:mask_positive-mask_negative,lambda:v)#二值网络
            newv=v.assign(prunev)
            newvs.append(newv)

        return  newvs

    def cout_ones(self):
        zeros_num=0.
        all_num=0.
        for v in tf.all_variables():
            zeros_num += tf.reduce_sum(tf.to_float(tf.less(tf.abs(v-tf.ones_like(v)), tf.ones_like(v)*0.0001)))#统计1的个数
            all_num += tf.reduce_sum(tf.ones_like(v))
        return [zeros_num,all_num]
        #梯度下降
    def optimer(self,loss,lr=0.1):
        grads=tf.gradients(loss, tf.all_variables())
        train_optimizers=[]
        for grad,v in zip(grads,tf.all_variables()):
            if grad is not None:
                train_optimizer =v.assign(v-grad*lr)
                train_optimizers.append(train_optimizer)


        return train_optimizers
class quantization(object):
    def __init__(self,number_cluster=10):
        with tf.variable_scope('codebook'):
            code=tf.get_variable("code_book",shape=(number_cluster,))
        onedims=[]
        for v in tf.all_variables():
            #with tf.variable_scope('codebook'):
            #    code=tf.get_variable(v.name.split(':')[0],v.get_shape().as_list()[-1])
            onedims.append(tf.squeeze(tf.reshape(v,(-1,1))))
            #self.codes.append(code)
        data=tf.concat(0,onedims)
        print data
   # def quantize(self):






