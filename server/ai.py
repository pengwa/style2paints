import tensorflow as tf
import keras
import numpy as np
from config import *
from keras.models import load_model


def ToGray(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    return 0.30 * R + 0.59 * G + 0.11 * B


def RGB2YUV(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 0.492 * (B - Y) + 128
    V = 0.877 * (R - Y) + 128
    return tf.concat([Y, U, V], axis=3)


def YUV2RGB(x):
    Y = x[:, :, :, 0:1]
    U = x[:, :, :, 1:2]
    V = x[:, :, :, 2:3]
    R = Y + 1.140 * (V - 128)
    G = Y - 0.394 * (U - 128) - 0.581 * (V - 128)
    B = Y + 2.032 * (U - 128)
    return tf.concat([R, G, B], axis=3)


def VGG2RGB(x):
    return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1]


print("---------00000000000000000-----")
# use 2.0.8 keara, otherwise, get_session() will fail due to graph is empty. don't know why.
session = keras.backend.get_session()

print("---------1111111111111111111111111111--------")
print(device_A + ",,,,,,," + device_B)
with tf.device(device_A):

    ipa = tf.placeholder(dtype=tf.float32, shape=(None, 1))
    ip1 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1))
    ip3 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
    ip4 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 4))
    ip3x = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

    baby = load_model('baby.net')
    baby_place = tf.concat([- 512 * tf.ones_like(ip4[:, :, :, 3:4]), 128 * tf.ones_like(ip4[:, :, :, 3:4]), 128 * tf.ones_like(ip4[:, :, :, 3:4])], axis=3)
    baby_yuv = RGB2YUV(ip4[:, :, :, 0:3])
    baby_alpha = tf.where(x=tf.zeros_like(ip4[:, :, :, 3:4]), y=tf.ones_like(ip4[:, :, :, 3:4]), condition=tf.less(ip4[:, :, :, 3:4], 128))
    baby_hint = baby_alpha * baby_yuv + (1 - baby_alpha) * baby_place
    baby_op = YUV2RGB(baby(tf.concat([ip1, baby_hint], axis=3)))
    baby_finder = tf.add(baby_op, baby_op, name="baby_finder")

    girder = load_model('girder.net')
    gird_op = (1 - girder([1 - ip1 / 255.0, ip4, 1 - ip3 / 255.0])) * 255.0
    gird_finder = tf.add(gird_op, gird_op, name="gird_finder")

    reader = load_model('reader.net')
    features = reader(ip3 / 255.0)
    featuresx = reader(ip3x / 255.0)

    head = load_model('head.net')
    feed = [1 - ip1 / 255.0, (ip4[:, :, :, 0:3] / 127.5 - 1) * ip4[:, :, :, 3:4] / 255.0]
    for _ in range(len(features)):
        item = keras.backend.mean(features[_], axis=[1, 2])
        itemx = keras.backend.mean(featuresx[_], axis=[1, 2])
        feed.append(item * ipa + itemx * (1 - ipa))
    nil0, nil1, head_temp = head(feed)

    neck = load_model('neck.net')
    nil2, nil3, neck_temp = neck(feed)
    feed[0] = tf.clip_by_value(1 - tf.image.resize_bilinear(ToGray(VGG2RGB(head_temp) / 255.0), tf.shape(ip1)[1:3]), 0.0, 1.0)
    nil4, nil5, head_temp = neck(feed)
    head_op = VGG2RGB(head_temp)
    head_finder = tf.add(head_op, head_op, name="head_finder")
    neck_op = VGG2RGB(neck_temp)
    neck_finder = tf.add(neck_op, neck_op, name="neck_finder")
    


print("---------2222222222222222222222--------")
with tf.device(device_B):

    ip3B = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

    tail = load_model('tail.net')
    pads = 7
    tail_op = tail(tf.pad(ip3B / 255.0, [[0, 0], [pads, pads], [pads, pads], [0, 0]], 'REFLECT'))[:, pads*2:-pads*2, pads*2:-pads*2, :] * 255.0
    tail_finder = tf.add(tail_op, tail_op, name="tail_finder")


print("---------333333333333333333333333333333333333333--------")
session.run(tf.global_variables_initializer())
tf.train.write_graph(session.graph,'.','style2paints.org.pbtxt', as_text=True)
tf.train.write_graph(session.graph,'.','style2paints.org.pb', as_text=False)

print("------------44444444444444444444444--------------------")
tail.load_weights('tail.net')
print("------------55555555555555555555555555--------------------")
baby.load_weights('baby.net')
print("------------666666666666666666666666--------------------")
head.load_weights('head.net')
print("------------7777777777777777777777777--------------------")
neck.load_weights('neck.net')
print("------------88888888888888888888888--------------------")
girder.load_weights('girder.net')
print("-----------9999999999999999999999--------------------")
reader.load_weights('reader.net')

print("------------00000000000000000000000000000000--------------------")

saver = tf.train.Saver()
saver.save(session, './model_ckt/my-model', global_step=0)

def go_head(sketch, global_hint, local_hint, global_hint_x, alpha):
    return session.run(head_op, feed_dict={
        ip1: sketch[None, :, :, None], ip3: global_hint[None, :, :, :], ip4: local_hint[None, :, :, :], ip3x: global_hint_x[None, :, :, :], ipa: np.array([alpha])[None, :]
    })[0].clip(0, 255).astype(np.uint8)


def go_neck(sketch, global_hint, local_hint, global_hint_x, alpha):
    return session.run(neck_op, feed_dict={
        ip1: sketch[None, :, :, None], ip3: global_hint[None, :, :, :], ip4: local_hint[None, :, :, :], ip3x: global_hint_x[None, :, :, :], ipa: np.array([alpha])[None, :]
    })[0].clip(0, 255).astype(np.uint8)


def go_gird(sketch, latent, hint):
    return session.run(gird_op, feed_dict={
        ip1: sketch[None, :, :, None], ip3: latent[None, :, :, :], ip4: hint[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


def go_tail(x):
    return session.run(tail_op, feed_dict={
        ip3B: x[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


def go_baby(sketch, local_hint):
    return session.run(baby_op, feed_dict={
        ip1: sketch[None, :, :, None], ip4: local_hint[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)

