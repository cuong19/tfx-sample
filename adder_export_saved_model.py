import tensorflow as tf


class Adder(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def add(self, x):
        return x + 1.


if __name__ == "__main__":
    to_export = Adder()
    tf.saved_model.save(to_export, 'tmp/adder/1')
