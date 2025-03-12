import matplotlib.pyplot as plt
import pylab as P
from scipy.io import wavfile as wav
import numpy as np
import IPython.display as ipd
rated, data = wav.read(r'D:\sss.wav')
print(rated, data.size)
#计算音频时长（总采样数/采样频率）
data.size/rated
#计算相邻样本的差值
e_signal = np.append(data[0], data[1:] - data[:-1])
print(e_signal.size)# 打印差分信号的样本数，验证其与原信号的样本数一致。
print(data[:10], e_signal[:10])
#根据差值重构出原信号
recompute = np.append(e_signal[0], e_signal[1:] + data[:-1])
print(np.sum(recompute-data))
#计算原信号和差值信号的方差
varSignal = np.var(data)
varE = np.var(e_signal)

print(varSignal, varE)
#显示原信号(蓝色'b'lue)和差值信号(红色'r'ed)
plt.plot(data, 'b')
plt.plot(e_signal, 'r')
plt.show()
#播放原信号和差值信号
ipd.display(ipd.Audio(data=data,rate=rated))
ipd.display(ipd.Audio(data=e_signal,rate=rated))


# 实现一个简单的DPCM一位编码函数
def DM_coding(input, delta):
    # 初始化
    x = np.zeros(input.shape)  # 表示编码值序列
    y = np.zeros(input.shape)  # 表示预测值序列
    x[0] = 1
    y[0] = 0

    n = input.size
    for i in range(1, n):
        # 预测值计算
        if x[i - 1] == 1:
            # 如果前一个编码输出是1,则当前样本的预测值等于前一个预测值加Δ
            y[i] = y[i - 1] + delta
        else:
            # 否则，当前样本的预测值等于前一个预测值减Δ
            y[i] = y[i - 1] - delta
        # 编码
        if input[i] > y[i]:
            # 如果实际样本值大于预测值
            x[i] = 1
        else:
            x[i] = 0

    return x
delta = 10
dm = DM_coding(data,delta)
plt.plot(dm[:100])
print(dm[:100])


# 定义一个DPCM解码函数
# def DM_decoding(x, delta):
#     # 初始化
#     input = np.zeros(x.shape)  # 表示原信号序列
#     # 请在此处插入你的代码.....
#
#     return input
# #仅根据1位DM编码结果进行音频的解码还原
# data2 = DM_decoding(dm, delta)
# plt.plot(data)
# plt.plot(data2*5,'r')
# ipd.display(ipd.Audio(data=np.array(data2, np.int16),rate=rated))