import  tensorflow as tf
import os
from google.protobuf import text_format
input_graph_def = tf.NodeDef()
with tf.gfile.FastGFile(os.path.join("model",'model_ori.ckpt'), 'rb') as f:
    input_graph_def.ParseFromString(f.read())
    #text_format.Merge(f.read().decode("utf-8"), input_graph_def)
