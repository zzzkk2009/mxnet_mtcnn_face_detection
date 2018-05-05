import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2

# detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker=2, accurate_landmark=False)
#
# print(detector.slice_index(5))


import numpy as np

A = np.arange(16)

A = A.reshape(2, 2, 4)
print('A.ndim=', A.ndim)
print('A.shape=', A.shape)
print('A.shape length=', len(A.shape))
print('A[0].ndim=', A[0].ndim)
print('A[0].shape=', A[0].shape)
print('A[0].shape length=', len(A[0].shape))
print('A[0][0].ndim=', A[0][0].ndim)
print('A[0][0].shape=', A[0][0].shape)
print('A[0][0].shape length=', len(A[0][0].shape))
print('A=', A)

print('A.transpose((0, 1, 2))=', A.transpose((0, 1, 2)))  # 保持不变
print('A.transpose((1, 0, 2))=', A.transpose((1, 0, 2)))  # 0轴和1轴交换，2轴保持不变

a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
c = np.vstack((a, b))
print('c=', c)
