"""
此文件里保存对data文件夹里的数据进行处理的程序
"""
# 导入需要的库
import csv
import numpy as np
import matplotlib.pylab as plt
from scipy import signal
from superlet import superlet

def remove_baseline(roi_list, f_sample, wp, ws, gpass, gstop):
    '''
    本函数使用butterworth带通滤波器去除基线漂移
    :param roi_list: 需要去除基线漂移的滤波器
    :param f_sample: 采样频率
    :param wp: 带通频率
    :param ws: 带阻频率
    :param gpass: 最大衰减量
    :param gstop: 最小衰减量
    :return: 滤波后的数组列表
    '''
    N, Wn = signal.buttord(wp, ws, gpass=1, gstop=40, fs=f_sample)  # 使用buttord函数计算滤波器阶数
    b, a = signal.butter(N, [wp, ws], btype='bandpass', output='ba', fs=f_sample)  # 计算滤波器函数
    filtered = signal.filtfilt(b, a, roi_list)
    return filtered

if __name__ == '__main__':
    with open('D:\\LUT\\IPPG-project-clone\\data\\vid_s1_T1_data.csv', 'r') as file_csv:  # 读取CSV文件
        reader = csv.reader(file_csv)
        for row in reader:
            roi_list = np.array(row, dtype=np.float64)

    f_sample = 30  # 设置采样率为30HZ
    wp = 0.5  # 设置带通频率
    ws = 3  # 设置截至频率
    filtered = remove_baseline(roi_list,f_sample,wp,ws,1,40)
    x = np.linspace(0,180,5997)
    plt.plot(x,filtered)
    plt.show()