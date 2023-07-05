"""
主函数流程
1.读取视频
2.将视频分解为帧
3.调用ROI识别函数
4.
"""
import numpy as np
import cv2
import dlib


path = 'D:/LUT/dateset/UBFC-PHYS/s1/vid_s1_T1.avi'  # 导入视频路径（注意不要出现\u）
cap = cv2.VideoCapture(path)  # 读取视频帧
cap.isOpened()
while cap.isOpened():
    ret, frame = cap.read()  # 从视频中提取视频帧
    frame_green = frame[:, :, 1]  # 将提取到的图像提取出绿色通道
    frame_green_deepcopy = frame_green
    frame_green_row = frame_green.shape[0]  # 读取绿色通道图像行数
    frame_green_col = frame_green.shape[1]  # 读取绿色通道图像列数
    img = frame
    predictor_model = "shape_predictor_68_face_landmarks.dat"  # 加载训练好的人脸检测数据
    detector = dlib.get_frontal_face_detector()  # 使用dlib库的人脸检测器
    predictor = dlib.shape_predictor(predictor_model)  # 用训练好的模型对人脸进行标定
    faces = detector(img, 1)  # 进行个数检测，默认参数为1
    landmarks = []  # 创建一个landmarks的数组
    for face in faces:  # for循环，检测多个人脸的特征点图像
        shape = predictor(img, face)
        for i in range(0, 68):
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks.append((x, y))
    landmarks = np.array(landmarks, dtype=np.uint32)
    zero_mask = np.full(frame.shape, 0)  # 创建掩膜大小和输入的图像相同，并增加一个通道，全部赋值为0
    if len(landmarks) != 68:  # 异常判断
        raise Exception('image_hull_mask works only with 68 landmask')
    inlandmarks = np.array(landmarks, dtype=np.int32)  # 创建特征点数组,将元组转变列表形式
    img = frame_green  # 读取灰度图像，进行灰度化
    hull_mask = np.full(img.shape, 0, dtype=np.int32)  # 创建掩膜大小和输入的图像相同，并增加一个通道，全部赋值为0
    # 创建一部字典包含感兴趣的ROI区域
    face_dcountry = {"left_cheek": np.vstack((inlandmarks[8:17], inlandmarks[27:31],
                                              inlandmarks[42:46], inlandmarks[54:55]))}
    left_cheek_landmark = np.array(face_dcountry['left_cheek'])
    hull_mask = cv2.fillPoly(hull_mask, [left_cheek_landmark], (255, 255, 255))  # 创建掩膜，进行人脸分割
    frame_green.astype(np.uint8)
    mask = hull_mask.astype(np.uint8)
    x = frame_green.shape
    y = hull_mask.shape
    z = type(frame_green)
    w = type(hull_mask)
    roi = cv2.bitwise_and(frame_green, frame_green, mask=mask)  # 对视频中提取到的绿色图像进行掩膜计算
    # for循环读取每一个像素的值，如果是0则忽略，若不为0则计数求平均值
    pixel_sum = 0
    pixel_count = 0
    for i in range(frame_green_row):
        for j in range(frame_green_col):
            pixel_value = frame_green[i, j]
            pixel_sum = pixel_sum + pixel_value
            if pixel_value != 0:
                pixel_count = pixel_count + 1
    roi_average = pixel_sum / pixel_count
    print("ROI区域像素平均值为：", roi_average)
cap.release()
cv2.destroyAllWindows()
