#coding=utf-8
import tensorflow as tf
import numpy as np
import time
import  matplotlib.pyplot as plt
#输入point必须是二维矩阵
def kmeans(points,ncluster):
    number,nfeature=points.get_shape().as_list()#nfeature为特征向量维度
    cluster_assignments = tf.Variable(tf.zeros([number], dtype=tf.int64))#用于保存最后的每个点聚类结果

    #slice函数，切分出前k个点，作为初始化聚类中心.
    #slice 函数可以用于切割子矩形图片，参数矩形框的rect,begin=(minx,miny),size=(width,height)
    centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [ncluster,nfeature]))


    #tile函数用于重复扩展，求取每个点到每个中心的距离，得到一个[number, ncluster]矩阵，用于保存每个点到聚类中心的距离
    rep_centroids = tf.reshape(tf.tile(centroids, [number, 1]), [number, ncluster, nfeature])
    rep_points = tf.reshape(tf.tile(points, [1, ncluster]), [number, ncluster, nfeature])
    sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),reduction_indices=2)



    #求取每个点的最近聚类中心
    best_centroids = tf.argmin(sum_squares, 1)
    #用于判断是否收敛的标识，tf.reduce_any(x)，如果矩阵x中有True，那么返回值就是True，也就是对矩阵x中的所有元素进行逻辑运算
    did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,cluster_assignments))



    def bucket_mean(data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    means = bucket_mean(points, best_centroids, ncluster)


    with tf.control_dependencies([did_assignments_change]):
        do_updates = tf.group(centroids.assign(means),cluster_assignments.assign(best_centroids))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:

        sess.run(init)
        changed = True
        iters = 0

        while changed and iters < 20:
            iters += 1
            [changed, _] = sess.run([did_assignments_change, do_updates])

            [points_np,centers, assignments] = sess.run([points,centroids, cluster_assignments])
    print 'kmeans is iter:%s'%iters
    return points_np,centers,assignments
#测试上面的k-means函数
def test():
    points = tf.Variable(tf.random_uniform([100,2]))
    points,centers,assigment=kmeans(points,5)

    for i,id in enumerate(assigment):
        if id==0:
            plt.plot(points[i,0],points[i,1],'ro')
        elif id==1:
            plt.plot(points[i,0],points[i,1],'go')
        elif id==2:
            plt.plot(points[i,0],points[i,1],'bo')
    plt.show()



