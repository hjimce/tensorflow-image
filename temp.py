import  tensorflow as tf
import  numpy as np

z=tf.placeholder(dtype=tf.float32,shape=[1000,1000])
x=tf.Variable(np.random.random(1000,1000),'x')
y=x*x*x*z
dz=tf.nn.l2_loss(tf.gradients(y,z))
dxz=tf.gradients(dz,x)
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    print session.run(dxz,feed_dict = {z:np.random.random((1000,1000))})

