

if __name__ == '__main__':
    import tensorflow as tf

    with tf.name_scope("test") as name_scope:
        print("name_scope: %s"%(name_scope))


    test1 = tf.truncated_normal([1,2],dtype=tf.float32)
    shape_of_test1 = tf.shape(test1)
    print("first shape_of_test1",shape_of_test1)
    print("type shape_of_test1", type(shape_of_test1))
    print(test1)
    print(type(test1))

    with tf.Session() as session:
        print("second shape of test1", session.run(shape_of_test1))
        print(session.run(test1))


    print("type of tf.Variable",type(tf.matmul))



    test1 = tf.placeholder(dtype=tf.float32,shape=[],name="test")
    test2 = tf.placeholder(dtype=tf.float32,shape=[],name="test")
    print("test1.name:",test1.name)
    print("test2.name:",test2.name)