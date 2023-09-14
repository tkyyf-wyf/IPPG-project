**python learning nodes**
# numpy库
## 如何创建一个numpy空数组
要使用 NumPy 库创建一个一维空数组，您可以使用 `numpy.empty()` 函数。以下是创建一维空数组的示例代码：

```python
import numpy as np

arr = np.empty(0)

print(arr)
```

在上述代码中，我们首先导入了 `numpy` 库，并使用 `np.empty()` 函数创建了一个一维空数组 `arr`。函数的参数是数组的形状，这里我们传入了 0，表示创建一个空数组。

最后，我们打印出数组 `arr` 的内容，您将看到一个空的一维数组。

请注意，`np.empty()` 函数创建的数组不会被初始化为特定的值，它只是分配了一块内存空间。数组的内容可能是随机的，取决于内存中的先前数据。如果您需要一个特定大小的数组，并且希望将其初始化为特定的值，可以使用 `np.zeros()` 或 `np.ones()` 函数。


# pycharm使用笔记
## debug
[pycharm官方使用手册](https://www.jetbrains.com/help/pycharm/using-breakpoints.html#breakpoint-statuses)

## 傅里叶变换及其逆变换
### 函数说明
1. fft(a, n = None, axis = -1, norm = None)
对一维数据进行变换.其中,a是n维数组(可实可虚);  
n为a数组的个数,小则截取,多则补0,n = None则表示所有的都会变换;  

2. fft2(a, s = None, axis = (-2,-1), norm = None)  
a上同fft;  
s 表示两轴或者多轴的情况,指定每个周上参与傅里叶变换的个数
3. fftn(a, n = None, axis = -1, norm = None)
