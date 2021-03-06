# coding: utf-8
# YuanYang
import math
import cv2
import numpy as np


def nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression
        非最大抑制，合并高度重合的候选窗口

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    return pick


def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = out_data.transpose((2, 0, 1))
    out_data = np.expand_dims(out_data, 0)
    out_data = (out_data - 127.5) * 0.0078125
    return out_data


def generate_bbox(map, reg, scale, threshold):
    """
        generate bbox from feature map
    Parameters:
    ----------
        map: numpy array , n x m x 1  ****但实际接收(xx, xx)
            detect score for each position
        reg: numpy array , n x m x 4  ****但实际接收(1, 4, xx, xx)
            bbox
        scale: float number
            scale of this detection
        threshold: float number
            detect threshold
    Returns:
    -------
        bbox array
    """
    stride = 2
    cellsize = 12

    t_index = np.where(map > threshold)
    # print('t_index')
    # print('t_index[0].shape=', t_index[0].shape)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])

    dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = map[t_index[0], t_index[1]]
    boundingbox = np.vstack([np.round((stride * t_index[1] + 1) / scale),
                             np.round((stride * t_index[0] + 1) / scale),
                             np.round((stride * t_index[1] + 1 + cellsize) / scale),
                             np.round((stride * t_index[0] + 1 + cellsize) / scale),
                             score,
                             reg])
    # print('boundingbox=', boundingbox)
    # print('boundingbox.shape=', boundingbox.shape)  # (9, xx)
    return boundingbox.T


def detect_first_stage(img, net, scale, threshold):
    """
        run PNet for first stage
    
    Parameters:
    ----------
        img: numpy array, bgr order
            input image
        scale: float number
            how much should the input image scale
        net: PNet
            worker
    Returns:
    -------
        total_boxes : bboxes
    """
    height, width, _ = img.shape
    hs = int(math.ceil(height * scale))
    ws = int(math.ceil(width * scale))

    im_data = cv2.resize(img, (ws, hs))

    # adjust for the network input
    input_buf = adjust_input(im_data)  # adjust the input from (h, w, c) to ( 1, c, h, w) for network input
    output = net.predict(input_buf)
    """
    候选人脸窗口
    output: 
    face classification: 1x1x2
    bounding box regression: 1x1x4
    facial landmark localization: 1x1x10
    """
    # print('output[0].shape=', output[0].shape)  # (1, 4, 99, 179) ...  (1, 4, 9, 19) ...
    # print('output[1].shape=', output[1].shape)  # (1, 2, 99, 179) ...  (1, 2, 9, 19) ...
    # print('output=', output)
    # print('output[1].shape=', output[1].shape)  # (1, 2, 2, 7)
    # print('output[1][0, 1, :, :]=', output[1][0, 1, :, :])
    # print('output[0]=', output[0])
    # print('output[1][0, 1, :, :]=', output[1][0, 1, :, :])
    # print('output[1][0, 1, :, :].shape=', output[1][0, 1, :, :].shape)
    # print('output[0].shape=', output[0].shape)
    # output[1][0, 1, :, :] 所有候选窗口位置
    # output[0] 边界框向量
    boxes = generate_bbox(output[1][0, 1, :, :], output[0], scale, threshold)

    if boxes.size == 0:
        return None

    # nms
    pick = nms(boxes[:, 0:5], 0.5, mode='Union')
    boxes = boxes[pick]
    return boxes


def detect_first_stage_warpper(args):
    return detect_first_stage(*args)
