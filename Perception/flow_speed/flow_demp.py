#coding:utf-8
import numpy as np
import cv2
import time  # 引入time模块

cap = cv2.VideoCapture('test2.mp4')

# ShiTomasi corner detection的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=20,
                      blockSize=2)
# 光流法参数
# maxLevel 未使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机生成的颜色
color = np.random.randint(0, 255, (100, 3))

y_offset = 150
x_offset = 300


# y_offset = 200
# x_offset = 30

ret, old_frame = cap.read()                             # 取出视频的第一帧
old_frame = old_frame[ y_offset : old_frame.shape[0] - y_offset, x_offset: old_frame.shape[1]-x_offset,:]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # 灰度化

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)                         # 为绘制创建掩码图片
ticks_last = time.time()                                # 记录时刻
y_sum_offset = 0                                        # 统计固定时间位移
while True:
    _, frame = cap.read()
    frame = frame[ y_offset : frame.shape[0] - y_offset, x_offset: frame.shape[1]-x_offset,:]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 计算光流以获取点的新位置
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选择good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # 绘制跟踪框
    y_sum = 0.0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        y_sum = y_sum + d - b                              # 计算两次特征点y轴方向位移和

    y_sum = y_sum / len(good_new)                          # 归一化，表明整体位移

    y_sum_offset = y_sum_offset + y_sum                    # 一定时间内位移和

    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    # cv2.imshow('gray',old_gray)
    k = cv2.waitKey(30)  # & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    # p0 = good_new.reshape(-1, 1, 2)
    ticks_now = time.time()                               # 当前时刻
    time_error = ticks_now - ticks_last                   # 时间差
    if time_error >1:                                     # 1s 间隔
        print("speed is : (%f),time_error:(%f) "%(y_sum_offset/time_error,time_error))     # 1s y轴的移动量
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)                # 更新特征点
        mask = np.zeros_like(old_frame)                         # 为绘制创建掩码图片
        ticks_last = ticks_now
        y_sum_offset = 0
    else:
        p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()