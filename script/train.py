import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('directory', 'data', """Directory where to read *.tfrecords.""")

SIZE_WIDTH = 28
SIZE_HEIGHT = 28

def convert_to(dataset, name):
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    # pip3 install pillow
    from PIL import Image

    for data in dataset:
        image_path = data[0]
        image_label = data[1]

        image = Image.open(image_path).convert("RGB").resize((SIZE_WIDTH, SIZE_HEIGHT))
        width, height = image.size

        record = tf.train.Example(features=tf.train.Features(feature={
            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[image_label])),
            'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'depth' : tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
        }))
        writer.write(record.SerializeToString())

    writer.close()

def inputs():
    if not FLAGS.directory:
        raise ValueError('Please supply a directory')

    tfrecords_filename = 'train'
    filename = os.path.join(FLAGS.directory, tfrecords_filename + '.tfrecords')
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)

    return image, label


def main():
    argv = sys.argv
    dataset = [
        ['./images/1.jpg', 0],
        ['./images/2.jpg', 1]
    ]

    convert_to(dataset, 'train')

if __name__ == '__main__':
    main()
