import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('directory', 'data', """Directory where to read *.tfrecords.""")

SIZE_WIDTH = 28
SIZE_HEIGHT = 28

def convert_to(dataset, filename):
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
            'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        }))
        writer.write(record.SerializeToString())

    writer.close()

def read_tensor_from(filename, input_mean=0, input_std=255):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
    })
    image = tf.image.decode_jpeg(features['image'], channels=3)
    float_image = tf.cast(image, tf.float32)
    dims_expander = tf.expand_dims(float_image, 0)

    resized = tf.image.resize_bilinear(dims_expander, [SIZE_HEIGHT, SIZE_WIDTH])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def main():
    argv = sys.argv
    dataset = [
        ['./images/1.jpg', 0],
        ['./images/2.jpg', 1]
    ]

    filename = os.path.join(FLAGS.directory, 'train.tfrecords')

    convert_to(dataset, filename)
    tensor = read_tensor_from('./images/input.jpg')
    print(tensor)

if __name__ == '__main__':
    main()
