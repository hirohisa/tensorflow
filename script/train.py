import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def main():
    argv = sys.argv

    img_data = [
        ['./images/1.jpg', 0],
        ['./images/2.jpg', 1]
    ]

    record_file = './test.tfrecords'
    writer = tf.python_io.TFRecordWriter(record_file)

    # pip3 install pillow
    from PIL import Image

    for img_datum in img_data:
        img_file = img_datum[0]
        img_label = img_datum[1]

        img = Image.open(img_file)
        width, height = img.size

        record = tf.train.Example(features=tf.train.Features(feature={
            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[img_label])),
            'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'depth' : tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
        }))
        writer.write(record.SerializeToString())

if __name__ == '__main__':
    main()
