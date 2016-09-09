#coding=utf-8
import tensorflow as tf
import numpy as np
import time

def kmeans(points,ncluster):
    number=points.get_shape().as_list()[0]
    cluster_assignments = tf.Variable(tf.zeros([number], dtype=tf.int64))#用于保存最后的聚类结果

    #slice函数，切分出前k个点，作为初始化聚类中心
    centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [ncluster,2]))

    rep_centroids = tf.reshape(tf.tile(centroids, [number, 1]), [number, ncluster, 2])
    rep_points = tf.reshape(tf.tile(points, [1, ncluster]), [number, ncluster, 2])
    sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),reduction_indices=2)



    #
    best_centroids = tf.argmin(sum_squares, 1)
    did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                    cluster_assignments))

    def bucket_mean(data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    means = bucket_mean(points, best_centroids, K)


    with tf.control_dependencies([did_assignments_change]):
        do_updates = tf.group(centroids.assign(means),cluster_assignments.assign(best_centroids))

    init = tf.initialize_all_variables()

