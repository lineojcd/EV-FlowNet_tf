import tensorflow as tf

# Read and see the contents of a TFRecord file

path = "/Users/jcd/PythonProjects/mydataset_event/outdoor_day1/left_event_images.tfrecord"

def peek_tfrecord(path, n=1):
    ds = tf.data.TFRecordDataset(path, compression_type=None)
    for i, raw in enumerate(ds.take(n)):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        print(f"--- Example #{i} ---")
        for k, v in ex.features.feature.items():
            # show a short summary
            if v.bytes_list.value:
                val = v.bytes_list.value[0]
                print(k, f"bytes_list len={len(v.bytes_list.value)} first_bytes={len(val)}")
            elif v.float_list.value:
                print(k, f"float_list len={len(v.float_list.value)} first={v.float_list.value[0]}")
            elif v.int64_list.value:
                print(k, f"int64_list len={len(v.int64_list.value)} first={v.int64_list.value[0]}")
        print()
        
def peek_sequence_example(path, n=1):
    ds = tf.data.TFRecordDataset(path)
    for i, raw in enumerate(ds.take(n)):
        ex = tf.train.SequenceExample()
        ex.ParseFromString(raw.numpy())
        print(f"--- SequenceExample #{i} ---")
        print("Context keys:", list(ex.context.feature.keys()))
        print("Feature lists keys:", list(ex.feature_lists.feature_list.keys()))

peek_tfrecord(path, n=2)
peek_sequence_example(path, n=1)
