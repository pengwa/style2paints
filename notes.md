freeze graph: 

pengwa@stcag-047:~/community/style2paints$ python3 -m tensorflow.python.tools.freeze_graph --input_graph=style2paints.org.pb --input_binary=true --output_node_names=strided_slice_20 --input_checkpoint=model_ckt/my-model-0 --output_graph=/tmp/frozen/style2paints_head.pb
/usr/local/lib/python3.5/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
2018-07-25 02:12:44.146098: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-07-25 02:12:44.387887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 40a8:00:00.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-07-25 02:12:44.583453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 1 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5ee9:00:00.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-07-25 02:12:44.776158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 2 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7cf5:00:00.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-07-25 02:12:44.976007: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 3 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: c2da:00:00.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-07-25 02:12:44.976219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0, 1, 2, 3
2018-07-25 02:12:46.342890: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-07-25 02:12:46.342978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 1 2 3
2018-07-25 02:12:46.342998: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N N N N
2018-07-25 02:12:46.343013: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 1:   N N N N
2018-07-25 02:12:46.343027: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 2:   N N N N
2018-07-25 02:12:46.343041: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 3:   N N N N
2018-07-25 02:12:46.343956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10755 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 40a8:00:00.0, compute capability: 3.7)
2018-07-25 02:12:46.578198: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10755 MB memory) -> physical GPU (device: 1, name: Tesla K80, pci bus id: 5ee9:00:00.0, compute capability: 3.7)
2018-07-25 02:12:46.811931: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 10755 MB memory) -> physical GPU (device: 2, name: Tesla K80, pci bus id: 7cf5:00:00.0, compute capability: 3.7)
2018-07-25 02:12:47.046204: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:3 with 10755 MB memory) -> physical GPU (device: 3, name: Tesla K80, pci bus id: c2da:00:00.0, compute capability: 3.7)
Converted 640 variables to const ops.

