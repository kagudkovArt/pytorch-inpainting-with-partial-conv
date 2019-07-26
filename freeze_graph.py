import tensorflow as tf


def frozen_graph_maker(export_dir,output_graph):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                sess.graph_def,
                output_nodes# The output node names are used to select the usefull nodes
        )
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())


export_dir = '/mnt/sdb/results/inpainting/models/1500000.pb'
output_graph = "/mnt/sdb/results/inpainting/models/frozen_graph.pb"
frozen_graph_maker(export_dir,output_graph)