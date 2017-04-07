#coding=utf-8
import  tensorflow  as tf
import  os
import  cv2

def load_model(session,netmodel_path,param_path):
    new_saver = tf.train.import_meta_graph(netmodel_path)
    new_saver.restore(session, param_path)
    x= tf.get_collection('test_images')[0]#在训练阶段需要调用tf.add_to_collection('test_images',test_images),保存之
    y = tf.get_collection("test_inf")[0]
    batch_size = tf.get_collection("batch_size")[0]
    return  x,y,batch_size

def load_images(data_root):
    filename_queue = tf.train.string_input_producer(data_root)
    image_reader = tf.WholeFileReader()
    key,image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    return image, key

def test(data_root="data/race/cropbrown"):
    image_filenames=os.listdir(data_root)
    image_filenames=[(data_root+'/'+i) for i in image_filenames]


    #print cv2.imread(image_filenames[0]).shape
    #image,key=load_images(image_filenames)
    race_listsrc=['black','brown','white','yellow']
    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)



        x,y,batch_size=load_model(session,os.path.join("model",'model_ori_race.ckpt.meta'),
                       os.path.join("model",'model_ori_race.ckpt'))
        predict_label=tf.cast(tf.argmax(y,1),tf.int32)
        print x.get_shape()
        for imgf in image_filenames:
            image=cv2.imread(imgf)
            image=cv2.resize(image,(76,76)).reshape((1,76,76,3))
            print "cv shape:",image.shape


            #cv2.imshow("t",image_np[:,:,::-1])
            y_np=session.run(predict_label,feed_dict = {x:image, batch_size:1})
            print race_listsrc[y_np]


        coord.request_stop()#queue需要关闭，否则报错
        coord.join(threads)
test()



