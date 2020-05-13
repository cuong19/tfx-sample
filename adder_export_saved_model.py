from collections import namedtuple

import tensorflow as tf
import mxnet as mx
import numpy as np
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

model_path= 'resnet50v2/resnet50-v2.onnx'
sym, arg_params, aux_params = import_model(model_path)

# Determine and set context
if len(mx.test_utils.list_gpus())==0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)
# Load module
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

Batch = namedtuple('Batch', ['data'])
def predict(img):
    mod.forward(Batch([img]))
    # Take softmax to generate probabilities
    scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
    # print the top-5 inferences class
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    i = a[0]
    return 'class=%s ; probability=%f' %(labels[i],scores[i])


class Adder(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def add(self, x):
        img = mx.nd.array(x)
        label = predict(img)
        return tf.constant(label, shape=(1,))


if __name__ == "__main__":
    to_export = Adder()
    tf.saved_model.save(to_export, 'tmp/adder/1')
