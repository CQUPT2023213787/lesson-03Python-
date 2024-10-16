# Python常用库学习 (1)——Numpy

  无论是搞数据分析还是人工智能。 选择 Python 这门语言的确可以让你相对快速地上车，而但靠裸 Python，它并不能支撑起这么多拥护者， 之所以 Python 被称为万能编程语言，正式因为它有很多实用，扎实的第三方库。

----

## Numpy库（Number + Python）

![numpy_logo](图片\numpy_logo.png)

**经常使用 Numpy 的场景：**

1. 需要批量处理数据的时候
2. 机器学习，人工智能这些需要进行海量数据运算处理的地方
3. 写游戏里面的物体运行逻辑时，经常涉及到矩阵、向量运算
4. 机器人模拟环境，背后的环境反馈信息，全是靠批量数据算出来的
5. 任何需要做统计的时候（爬虫爬完了信息后）
6. 画图表之前，要对数据做一轮批量处理



---

### 安装Numpy库

使用Anaconda promote安装Numpy库

![](图片\20BC47B17F96052F2812549A2E8C7AFA.png)

`pip` 的方法很简单，你只需要在终端里面输入下面这样：

```
pip install numpy
```

如果你是 Python3.+ 的版本，用下面这种方式：

```
pip3 install numpy
```

怎么确认自己已经安装好了？首先，它打印出正确安装的信息，然后你再输入这句话，如果没有提示任何信息，则安装好。

```
python3 -c "import numpy"
```

如果提示下面这样的信息，就意味着你的安装失败，请再尝试一下前面的流程。

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'numpy'
```



---

### 写Numpy程序

当你安装好了，你就可以在自己的文件中写 Numpy 代码了。一般的流程是你先 `import numpy`。为了后续调用 `numpy` 更方便，我们通常在 `import` 完之后， 还给它一个缩写形式，`as np`。接下来你就能用 `np.xxx` 写 Numpy 的代码了

```python
import numpy as np

print(np.array([1,2,3]))
```



---

### Numpy Array

在 Numpy 中，我们会一直使用到它的一种 Array 数据。这个 Array 的格式和形态多种多样，我们会在[后续的教程](https://mofanpy.com/tutorials/data-manipulation/numpy/ndarray)中更详细的介绍。 现在，你只需要懂得如何 `import numpy`，并像下面这样定义一个 `array` 就好了。

```python
import numpy as np

np.array([1,2,3])
```

其实提到 Numpy，我建议你多做一些和 Python 原生 [List](https://mofanpy.com/tutorials/python-basic/interactive-python/data) 的对比， 因为它们从结构和形式上来说，是十分类似的。

了解 Python 原生 [List](https://mofanpy.com/tutorials/python-basic/interactive-python/data) 的朋友一定都知道，当你想要存储一些数据的时候， 你很可能想要用一个 list 来存储，并且可以按顺序提取出来。比如下面这样:

```python
my_list = [1,2,3]
print(my_list[0])
```

存储和提取就是 `List` 的最基本用法和功能。而 Numpy `Array` 也能做这件事。

```python
my_array = np.array([1,2,3])
print(my_array[0])
```

这里小伙伴肯定会疑惑，相较于Python List，那Numpy的优势体现在哪呢？

**Numpy的核心优势：运算快**。用专业的语言描述的话，`Numpy` 喜欢用电脑内存中连续的一块物理地址存储数据，因为都是连号的嘛，找到前后的号，不用跑很远， 非常迅速。而 `Python` 的 `List` 并不是连续存储的，它的数据是分散在不同的物理空间，在批量计算的时候，连号的肯定比不连号的算起来更快。因为找他们的时间更少了。

![](图片\numpy_list.png)

而且 Numpy Array 存储的数据格式也有限制，尽量都是同一种数据格式，这样也有利于批量的数据计算。 所以只要是处理大规模数据的批量计算，`Numpy` 肯定会比 `Python` 的原生 `List` 要快。

下面我们用例子来展示一下两者的速度对比：

```python
import time

t0 = time.time()
# python list
l = list(range(100))
for _ in range(10000):
    for i in range(len(l)):
        l[i] += 1

t1 = time.time()
# numpy array
a = np.array(l)
for _ in range(10000):
    a += 1

print("Python list spend {:.3f}s".format(t1-t0))
print("Numpy array spend {:.3f}s".format(time.time()-t1))

```

work:

```Python list spend 0.129s
Python list spend 0.129s
Numpy array spend 0.012s
```

果然，比起Python list，Numpy运算速度是杠杠的~



---

#### 数组的创建

创建数组有多种方式。你可以使用`np.array`直接用Python的元组和列表来创建。

```python
import numpy as np
a=np.array([1,2,3])
print(a.dtype)
b=np.array([1.1,2.2,3.3])
print(b.dtype)
c=np.array([(1,2,3),(4.5,5,6)]) #创建二维数组
print(c)
d=np.array([(1,2),(3,4)],dtype=complex) #数组的类型可以在创建时显式声明
print(d)
```

work:

```
int32
float64
[[ 1.   2.   3. ]
 [ 4.5  5.   6. ]]
[[ 1.+0.j  2.+0.j]
 [ 3.+0.j  4.+0.j]]
```

通常，数组的元素的未知的，但是形状确实已知的。所以NumPy提供了多种创建空数组的方法。
`np.zeros` 创建全是0的数组。
`np.ones` 创建全是1的数组。
`np.empty` 创建初始值是随机数的数组。
需要注意的是上述方法创建的数组元素的类型是 `float64`

```python
e=np.zeros((3,4))
print(e)
f=np.ones((2,3,4),dtype=np.int16)#可以更改数据类型
print(f)
g=np.empty((2,3))
print(g)
```

work:

```
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
[[[1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]]

 [[1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]]]
[[ 1.   2.   3. ]
 [ 4.5  5.   6. ]]
```

为了创建列表，NumPy提供了和 `range` 类似的函数。
`np.arange(start,end,step)`

```python
a=np.arange(10,30,5)
print(a)
b=np.arange(0,2,0.3)#同样可以接收浮点数
print(b)
```

work:

```
[10 15 20 25]
[ 0.   0.3  0.6  0.9  1.2  1.5  1.8]
```

注意：

在生成浮点数列表时，最好不要使用`np.arange`，而是使用`np.linspace`。
`np.linspace(start,stop,num)`

```python
np.linspace(0,2,9)
```

work：

```
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
```



---

#### 打印数组

当你打印一个数组时，NumPy显示数组的方式和嵌套的列表类似，但是会遵循以下布局：

- 最后一维从左到右显示
- 第二维到最后一维从上到下显示
- 剩下的同样从上到下显示，以空行分隔

一维数组显示成一行，二维数组显示成矩阵，三维数组显示成矩阵的列表。

```python
a=np.arange(6)
print(a)
b=np.arange(12).reshape(4,3)
print(b)
c=np.arange(24).reshape(2,3,4)
print(c)
```

work：

```
[0 1 2 3 4 5]
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

当一个数组元素太多，不方便显示时，NumPy会自动数组的中间部分，只显示边角的数据。

```python
print(np.arange(10000))
```

work:

```
[   0    1    2 ..., 9997 9998 9999]
```



---

#### 基本操作

数组的算数计算是在元素层级运算的。计算结果会存在一个新创建的数组中。

```python
import numpy as np
a=np.array([20,30,40,50])
b=np.arange(4)
print(b)
c=a-b
print(c)
print(b**2)
print(10*np.sin(a))
print(a<35)
```

work:

```
[0 1 2 3]
[20 29 38 47]
[0 1 4 9]
[ 9.12945251 -9.88031624  7.4511316  -2.62374854]
[ True  True False False]
```



在NumPy中`*`号仍然表示乘法，矩阵乘积用`np.dot`来计算。

```python
A=np.array([(1,1),(0,1)])
B=np.array([(2,0),(3,4)])
print(A*B)
print(A.dot(B))
print(np.dot(A,B))
```

work:

```
[[2 0]
 [0 4]]
[[5 4]
 [3 4]]
[[5 4]
 [3 4]]
```



类似于`+=`和`*=`的运算是直接在现有数组上计算的，没有创建新的数组。Numpy中的计算同样也是向上转型的，可以简单理解成浮点数和整数运算的结果是浮点数。

```python
a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
a*=3
print(a)
b += a
print(b)
# a += b                  # 浮点数不会自动转换成整数
```

work:

```
[[3 3 3]
 [3 3 3]]
[[ 3.36167598  3.63342297  3.22543331]
 [ 3.17992397  3.01462584  3.87847828]]
```



`np.ndarray`提供了许多一元操作。比如数组求和、求最大最小值等。

```python
a=np.random.random((2,3))
print(a)
print(a.sum())
print(a.mean())
print(a.max())
print(a.min())
```

work:

```
[[ 0.06108727  0.21625055  0.066292  ]
 [ 0.20271722  0.93946432  0.37747181]]
1.86328317161
0.310547195269
0.939464322779
0.0610872663968
```



默认的，这些一元操作是对整个数组进行计算，没有考虑到数组的形状。你可以设置`axis`参数来指定运算方向。`axis`表示第n维（从0开始）。

```python
b=np.arange(12).reshape(3,4)
print(b)
print(b.sum(axis=0)) #对第0维的元素求和
print(b.sum(axis=1)) #对第1维的元素求和
print(b.min(axis=1))
print(b.cumsum(axis=1)) #对第1维的元素累加求和
```

work:

```
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[12 15 18 21]
[ 6 22 38]
[0 4 8]
[[ 0  1  3  6]
 [ 4  9 15 22]
 [ 8 17 27 38]]
```



---

#### 广播函数

NumPy提供了熟知的数学方法，如：sin、cos、exp等。在NumPy中，这些方法被称作广播函数。这些函数会对数组中的每个元素进行计算，返回计算后的数组。

```python
B=np.arange(3)
print(B)
print(np.exp(B))
print(np.sqrt(B))
C=np.array([2,-1,4])
print(np.add(B,C))
print(B+C)
```

work:

```
[0 1 2]
[ 1.          2.71828183  7.3890561 ]
[ 0.          1.          1.41421356]
[2 0 6]
[2 0 6]
```



---

#### 索引、切片和迭代

一维数组可以被索引、切片和迭代，就和Python中的列表一样。

```python
a=np.arange(10)**3
print(a)
print(a[2])
print(a[2:5])
a[:6:2]=-1000
print(a)
print(a[::-1])
for i in a:
    print(i)
```

work:

```
[  0   1   8  27  64 125 216 343 512 729]
8
[ 8 27 64]
[-1000     1 -1000    27 -1000   125   216   343   512   729]
[  729   512   343   216   125 -1000    27 -1000     1 -1000]
-1000
1
-1000
27
-1000
125
216
343
512
729
```

多维数组可以在每一个维度有一个索引，这些索引构成元组来进行访问。

```python
def f(x,y):return 10*x+y
b=np.fromfunction(f,(5,4),dtype=int)
print(b)
print(b[2,3])
print(b[0:5,1])
print(b[:,1])
print(b[1:3,:])
```

work:

```
[[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]
23
[ 1 11 21 31 41]
[ 1 11 21 31 41]
[[10 11 12 13]
 [20 21 22 23]]
```

`...`表示对索引的省略。如下所示：

```python
c = np.array( [[[  0,  1,  2],               # 三维数组
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])
print(c.shape)
print(c[1,...])                                   # 和 c[1,:,:] 、 c[1]效果相同
print(c[...,2])                                   # 和c[:,:,2]效果相同
```

work:

```
(2, 2, 3)
[[100 101 102]
 [110 112 113]]
[[  2  13]
 [102 113]]
```



```
对多维数组的迭代是在第一维进行迭代的。
```

```python
for row in b:
    print(row)
```

work:

```
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```

如果需要遍历多维数组的所有元素，可以使用`flat`这个属性。

```python
for element in b.flat:
    print(element)
```

work:

```
0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43
```



---

### 数组形状操作

#### 更改数组的形状

有很多种方式可以更改数组的形状。下列的函数都没有对原数组进行更改，而是返回了一个更改后的新数组。

```python
a = np.floor(10*np.random.random((3,4)))
print(a.ravel()) #返回铺平后的数组
print(a.reshape(6,2)) #按照指定的形状更改
print(a.T)#返回转置矩阵
```

work:

```
[ 5.  0.  9.  5.  5.  4.  2.  2.  3.  2.  0.  7.]
[[ 5.  0.]
 [ 9.  5.]
 [ 5.  4.]
 [ 2.  2.]
 [ 3.  2.]
 [ 0.  7.]]
[[ 5.  5.  3.]
 [ 0.  4.  2.]
 [ 9.  2.  0.]
 [ 5.  2.  7.]]
```

如果一个维度填的是-1，则该维度的形状会自动进行计算

```python
print(a.reshape(3,-1))
```

work:

```
[[ 5.  0.  9.  5.]
 [ 5.  4.  2.  2.]
 [ 3.  2.  0.  7.]]
```

---

#### 堆砌不同的数组

多个数组可以按照不同的轴合在一起

```python
a=np.floor(10*np.random.random((2,2)))
print(a)
b=np.floor(10*np.random.random((2,2)))
print(b)
print(np.vstack((a,b)))#垂直方向堆砌
print(np.hstack((a,b)))#水平方向堆砌
from numpy import newaxis
print(a[:,newaxis])
```

work:

```
[[ 5.  1.]
 [ 4.  2.]]
[[ 8.  1.]
 [ 7.  8.]]
[[ 5.  1.]
 [ 4.  2.]
 [ 8.  1.]
 [ 7.  8.]]
[[ 5.  1.  8.  1.]
 [ 4.  2.  7.  8.]]
[[[ 5.  1.]]

 [[ 4.  2.]]]
```



---

#### 将一个数组划分为多个更小的数组

使用`hsplit`，`vsplit`可以对数组按照水平方向和垂直方向进行划分。

```python
a=np.floor(10*np.random.random((2,12)))
print(a)
print(np.hsplit(a,3))
print(np.hsplit(a,(1,2,3)))#在第一列，第二列，第三列进行划分
```

work:

```
[[ 7.  4.  0.  7.  5.  6.  4.  4.  4.  7.  7.  0.]
 [ 0.  1.  7.  7.  4.  9.  7.  0.  0.  2.  7.  5.]]
[array([[ 7.,  4.,  0.,  7.],
       [ 0.,  1.,  7.,  7.]]), array([[ 5.,  6.,  4.,  4.],
       [ 4.,  9.,  7.,  0.]]), array([[ 4.,  7.,  7.,  0.],
       [ 0.,  2.,  7.,  5.]])]
[array([[ 7.],
       [ 0.]]), array([[ 4.],
       [ 1.]]), array([[ 0.],
       [ 7.]]), array([[ 7.,  5.,  6.,  4.,  4.,  4.,  7.,  7.,  0.],
       [ 7.,  4.,  9.,  7.,  0.,  0.,  2.,  7.,  5.]])]
```



---

### 复制和视图

当操作数组时，数组的数据有时会复制到新数组中，有时又不会。这通常令初学者感到困难。总的来说有下面三种情况：

#### 不复制

简单的赋值不会复制数组的数据。

```python
a=np.arange(12)
b=a
print(b is a)
b.shape=3,4
print(a.shape)
```

work:

```
True
(3, 4)
```

---

#### 视图和浅复制

不同的数组可以使用同一份数据，`view`函数在同一份数据上创建了新的数组对象。

```python
c=a.view()
print(c is a)
print(c.base is a) #c是a的数据的视图
print(c.flags.owndata)
c.shape=6,2
print(a.shape) #a的形状没有改变
c[4,1]=1234 #a的数据改变了
print(a)
```

work:

```
False
True
False
(3, 4)
[[   0    1    2    3]
 [   4    5    6    7]
 [   8 1234   10   11]]
```

对数组切片会返回数组的视图

```python
s=a[:,1:3]
s[:]=10
print(a)
```

work:

```
[[ 0 10 10  3]
 [ 4 10 10  7]
 [ 8 10 10 11]]
```

---

#### 深复制

`copy`函数实现了对数据和数组的完全复制。

```python
d=a.copy()
print(d is a)
print(d.base is a)
d[0,0]=9999
print(a)
```

work:

```
False
False
[[ 0 10 10  3]
 [ 4 10 10  7]
 [ 8 10 10 11]]
```



---

### 多种多样的索引和索引的小技巧

相比Python的列表，NumPy提供了更多的索引功能。除了可以用整数和列表来访问数组之外，数组还可以被整型数组和布尔数组访问。

#### 用数组访问数组

```python
a=np.arange(12)**2
i=np.array([1,1,3,8,5])
print(a[i])
j=np.array([[3,4],[8,5]]) #用二维数组来访问数组
print(a[j]) #产生和访问的数组相同形状的结果
```

work:

```
[ 1  1  9 64 25]
[[ 9 16]
 [64 25]]
```



在时间序列的数据上寻找最大值通常会用到数组索引

```python
time=np.linspace(20,145,5)
data=np.sin(np.arange(20)).reshape(5,4)
print(time)
print(data)
ind=data.argmax(axis=0)#返回按照指定轴的方向的最大值的索引
time_max=time[ind]
print(time_max)
data_max=data[ind,range(data.shape[1])]
print(data_max)
```

work:

```
[  20.     51.25   82.5   113.75  145.  ]
[[ 0.          0.84147098  0.90929743  0.14112001]
 [-0.7568025  -0.95892427 -0.2794155   0.6569866 ]
 [ 0.98935825  0.41211849 -0.54402111 -0.99999021]
 [-0.53657292  0.42016704  0.99060736  0.65028784]
 [-0.28790332 -0.96139749 -0.75098725  0.14987721]]
[  82.5    20.    113.75   51.25]
[ 0.98935825  0.84147098  0.99060736  0.6569866 ]
```



你也可以使用数组索引来赋值

```python
a=np.arange(5)
a[[1,3,4]]=0
print(a)
```

work:

```
[0 0 2 0 0]
```

如果赋值时有重复的索引，则赋值会执行多次，留下最后一次执行的结果

```
a=np.arange(5)
a[[0,0,0]]=[1,2,3]
print(a)
```

work:

```
[3 1 2 3 4]
```

但是赋值时使用`+=`时，并不会重复计算

```python
a=np.arange(5)
a[[0,0,0]]+=1
print(a)
```

work:

```
[1 1 2 3 4]
```

这是因为"a+=1"最终是解释成了"a=a+1"



---

#### 用布尔数组来访问数组

通过使用布尔数组索引，我们可以选择哪些数据是需要的，哪些是不需要的。
在赋值中也非常有用。

```python
a = np.arange(12).reshape(3,4)
b = a > 4
print(b)
print(a[b])
a[b]=10
print(a)
```

work:

```
[[False False False False]
 [False  True  True  True]
 [ True  True  True  True]]
[ 5  6  7  8  9 10 11]
[[ 0  1  2  3]
 [ 4 10 10 10]
 [10 10 10 10]]
```



---

#### ix_()函数

ix_函数被用来计算不同的向量的乘积。

```python
a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])
ax,bx,cx = np.ix_(a,b,c)
print(ax)
print(bx)
print(cx)
print(ax.shape, bx.shape, cx.shape)
result = ax*bx*cx + ax
print(result)
print(result[3,2,4])
print(a[3]*b[2]*c[4]+a[3])#计算的结果是相同的
```

work:

```
[[[2]]

 [[3]]

 [[4]]

 [[5]]]
[[[8]
  [5]
  [4]]]
[[[5 4 6 8 3]]]
(4, 1, 1) (1, 3, 1) (1, 1, 5)
[[[ 82  66  98 130  50]
  [ 52  42  62  82  32]
  [ 42  34  50  66  26]]

 [[123  99 147 195  75]
  [ 78  63  93 123  48]
  [ 63  51  75  99  39]]

 [[164 132 196 260 100]
  [104  84 124 164  64]
  [ 84  68 100 132  52]]

 [[205 165 245 325 125]
  [130 105 155 205  80]
  [105  85 125 165  65]]]
65
65
```



---

### 线性代数

提供基本的线性代数操作

#### 简单的数组操作

```python
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
a.transpose()
np.linalg.inv(a)
u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
j = np.array([[0.0, -1.0], [1.0, 0.0]])
np.dot (j, j) # 点积
np.trace(u)  # 矩阵的迹
y = np.array([[5.], [7.]])
print(np.linalg.solve(a, y))#解线性方程组
print(np.linalg.eig(j))#计算特征值
```

work:

```
[[ 1.  2.]
 [ 3.  4.]]
[[-3.]
 [ 4.]]
(array([ 0.+1.j,  0.-1.j]), array([[ 0.70710678+0.j        ,  0.70710678-0.j        ],
       [ 0.00000000-0.70710678j,  0.00000000+0.70710678j]]))
```





---



# Python常用库学习（1）——Pandas

大数据虽然描述的是海量的数据，但是大数据离你却并不远，特别是大数据所涵盖的技术，在你生活当中，是时刻都能使用这些大数据涉及到的技术， 来解决你生活中的具体问题。

![](图片\big_data.jpg)

其实当你也有想解决的数据问题，不管是一份考题，还是工作总结，拥有了这种处理数据的能力后，不光是你自己，就可能连身边的人都会受益于你的能力。

今天要讲的是 Pandas能解决以下问题：

- 办公自动化
  - 有 Excel 或者格式化的文本文件，需要进行数据加工处理
  - 对大量的这些文本文件作图，想要自动化处理
- 人工智能
  - 数据分析，可视化数据规律
  - 数据前处理，为 AI 模型展平道路

## Pandas 和 Numpy 的差别

### 类比Python来对比两者：

用过 Python，你肯定熟悉里面的**List 和 Dictionary**, 我比较常拿这两种形态来对比 Numpy 和 Pandas 的关系。

```python
a_list = [1,2,3]
a_dict = {"a": 1, "b": 2, "c": 3}
print("list:", a_list)
print("dict:", a_dict)
```

work:

```list: [1, 2, 3]
list: [1, 2, 3]
dict: {'a': 1, 'b': 2, 'c': 3}
```

上面就是一种最常见的 Python 列表和字典表达方式。而下面，我们展示的就是 Numpy 和 Pandas 的一种构建方式。

```python
import pandas as pd
import numpy as np

a_array = np.array([
    [1,2],
    [3,4]
])
a_df = pd.DataFrame(
    {"a": [1,3], 
     "b": [2,4]}
)

print("numpy array:\n", a_array)
print("\npandas df:\n", a_df)
```

work:

```
numpy array:
 [[1 2]
 [3 4]]

pandas df:
    a  b
0  1  2
1  3  4
```

你会发现，我们看到的结果中，Numpy 的是没有任何数据标签信息的，你可以认为它是纯数据。而 Pandas 就像字典一样，还记录着数据的外围信息， 比如标签（Column 名）和索引（Row index）。 这也是我为什么总说 Numpy 是 Python 里的列表，而 Pandas 是 Python 里的字典。

### 对比 Numpy:

在 **Numpy** 中， 如果你不特别在其他地方标注，你是不清楚记录的这里边记录的是什么信息的。**而 Pandas 记录的信息可以特别丰富， 你给别人使用传播数据的时，这些信息也会一起传递过去。或者你自己处理数据时对照着信息来加工数据，也会更加友善。**

这就是在我看来 Pandas 对比 Numpy 的一个最直观的好处。

**另外 Pandas 用于处理数据的功能也比较多，信息种类也更丰富，特别是你有一些包含字符的表格，Pandas 可以帮你处理分析这些字符型的数据表。 当然还有很多其它功能，比如处理丢失信息，多种合并数据方式，读取和保存为更可读的形式等等。**

这些都让 Pandas 绽放光彩。**但是，Pandas 也有不足的地方：运算速度稍微比 Numpy 慢。**

###  总结：

Pandas 是 Numpy 的封装库，继承了 Numpy 的很多优良传统，也具备丰富的功能组件，但是你还是得分情况来酌情选择要使用的工具。



---

## 从文件读取数据

### Excel文件

现在就让我们看看你最可能遇见的 Excel 文件是怎么用 Pandas 读取出来。我们先看比较简单的文件格式，用 Excel 打开是这样：

![](图片\excel_data.png)

你只需要使用使用 `pd.read_excel()` 就能读出来了。

```python
import pandas as pd
df = pd.read_excel("data/体检数据.xlsx", index_col=0)
df
```

work:

```
    姓名   身高  体重   肺活量
学号                   
1   小明  168  60  3200
2   小黄  187  80  3800
3   小花  170  70  3400
```

另外，我在 `pd.read_excel()` 当中使用了 `index_col=0` 这个参数，你先看看如果不使用这个，会显示什么。

````python
pd.read_excel("data/体检数据.xlsx")

````

work:

```
   学号  姓名   身高  体重   肺活量
0   1  小明  168  60  3200
1   2  小黄  187  80  3800
2   3  小花  170  70  3400
```

你看，前面还多了一列 `Unnammed：0`，使用 `index_col=0` 就是告诉 Pandas，让它使用第一个 column（学号）的数据当做 row 索引。 后面还有很多读取的功能里也有一样的参数。

好，我们既然可以读取 Excel 文件，那么稍稍修改，再保存起来应该也不成问题。

```python
df.loc[2, "体重"] = 1
df.to_excel("data/体检数据_修改.xlsx")
pd.read_excel("data/体检数据_修改.xlsx", index_col=0)
```

work:

```
    姓名   身高  体重   肺活量
学号                   
1   小明  168  60  3200
2   小黄  187   1  3800
3   小花  170  70  3400
```

其实在读取和保存 Excel 文件的时候，还有很多额外的参数可供选择，因为太多了，我们这里就先讲最常用的，如果你要深入研究， 可以到他们的官网来看[官方文档]（https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas.read_excel）



---

### csv或txt等纯文本文件

  txt，csv，log 等这些都可以是纯文本文件，不过值得注意的是，对于 Pandas，它只对结构化的纯文本文件感兴趣。如果在你的纯文本文件中， 不是用一些标准的分隔符来分割数据，那么 Pandas 也拿它无能为力，是解析不出来的。

  先看看这个**体检数据.csv文件**，你可以用 Excel 或者是 txt 编辑器打开它，我建议你还是用纯文本（txt）编辑器打开， 这样你就能观看到它最原始的样貌了。

![](图片\csv_data.png)

  上图就是我用纯文本打开后的样子，可见，它就是用逗号隔开的一些数据而已。你也能用 Python 的 open 打开纯文本。

```python
with open("data/体检数据.csv", "r", encoding="utf-8") as f:
    print(f.read())
```

work:

```
学号,姓名,身高,体重,肺活量
1,小明,168,60,3200
2,小黄,187,80,3800
3,小花,170,70,3400
```

  有的时候，你不能保证别人给你的数据，是不是一份标准格式的数据，比如别人不喜欢用 `,` 来分隔数据点， 而是喜欢用什么乱七八糟的 `=` 来分隔。这时，Pandas 帮你考虑到了这种问题， 你可以挑选要用哪个字符来识别这些分隔。

```python
with open("data/体检数据_sep.csv", "r", encoding="utf-8") as f:
    print(f.read())
df_csv = pd.read_csv("data/体检数据_sep.csv", index_col=0, sep="=")
df_csv
```

work:

```
学号=姓名=身高=体重=肺活量
1=小明=168=60=3200
2=小黄=187=80=3800
3=小花=170=70=3400

    姓名   身高  体重   肺活量
学号                   
1   小明  168  60  3200
2   小黄  187  80  3800
3   小花  170  70  3400
```

  提到 csv，你可能还会想用 Excel 打开看看，但是提到 txt，一般你也不会想用 Excel 打开了吧。用 Pandas 打开一个 txt 文件和打开一个 csv 文件，、 其实本质上是一样的，都是打开一个纯文本文件。所以下面我再打开一下 txt。

```python
with open("data/体检数据_sep.txt", "r", encoding="utf-8") as f:
    print(f.read())
df_txt = pd.read_csv("data/体检数据_sep.txt", index_col=0, sep="=")
df_txt

```

work:

```
学号=姓名=身高=体重=肺活量
1=小明=168=60=3200
2=小黄=187=80=3800
3=小花=170=70=3400

    姓名   身高  体重   肺活量
学号                   
1   小明  168  60  3200
2   小黄  187  80  3800
3   小花  170  70  3400
```

   能打开，我们就能保存，保存方法同样很简单，只需要 `df.to_csv()` 就好了，甚至，你还能保存到 Excel 文件，在 Pandas 中它们是可以互相转换的。 同理用 `read_excel()` 打开的，也能存成 `to_csv()`。

```python
df_txt.to_csv("data/体检数据_sep_修改.csv")
df_txt.to_excel("data/体检数据_sep_修改.xlsx")

print("读保存后的 csv")
print(pd.read_csv("data/体检数据_sep_修改.csv"))

print("读保存后的 xlsx")
print(pd.read_excel("data/体检数据_sep_修改.xlsx"))
```

好了，做数据分析和机器学习，会用上面的方法来读 Excel 或者是纯文本，我们就已经解决了大部分的需求了。



---

## Pandas 中的数据格式

  简单来说，Pandas 支持最好的是一维和二维数据，一维数据就是一个序列，一条数据，而二维数据是我们生活中更常见的种类，基本上所有 Excel 数据， 都是二维数据，有横纵交替，用两个维度来定位这个数据。

**Pandas **中的一维二维数据特性。会要涉及到的功能包括：

- 数据序列Series
  - 创建
  - 转换 Numpy
- 数据表DataFrame
  - 创建
  - 转换 Numpy

### 数据序列Series

一串 Python List 的形式你肯定不陌生，Pandas 中的 Series 的核心其实就是一串类似于 Python List 的序列。只是它要比 Python List 丰富很多， 有更多的功能属性。

```python
import pandas as pd

l = [11,22,33]
s = pd.Series(l)
print("list:", l)
print("series:", s)

```

work:

```
list: [11, 22, 33]
series: 0    11
1    22
2    33
dtype: int64
```

  打印出来，对比很明显，Pandas Series 还帮我们额外维护了一份索引。有这个索引有啥意义呢？Python List 不也有一个隐藏的序号索引吗？ 其实，Pandas 之所以做这一种索引，目的并不是仅让你用 0123 这样的序号来检索数据，它还想让你可以用自己喜欢的索引来检索。看看下面的代码吧。

```python
s = pd.Series(l, index=["a", "b", "c"])
s
```

work:

```
a    11
b    22
c    33
dtype: int64
```

  也就是说还能自定义索引。换个思路，是不是只要是有索引形式的结构，都可以搞成 Series？比如下面这样

```python
s = pd.Series({"a": 11, "b": 22, "c": 33})
s
```

work:

```
a    11
b    22
c    33
dtype: int64
```

  太神奇了吧，原来字典也可以变成一个序列！更神奇的还在后面的 **DataFrame**呢，也可以用字典来创建 DataFrame。

  既然 Python 的 List 可以用来创建 Series，那我想 Numpy 应该也可以吧，要不来试试。

```python
import numpy as np

s = pd.Series(np.random.rand(3), index=["a", "b", "c"])
s
```

work:

```
a    0.292454
b    0.693788
c    0.058512
dtype: float64
```

既然 Numpy 和 List 可以用来创建 Series，那 Series 能回退到 Numpy array 或者 List 吗? 试一试就知道。

```python
print("array:", s.to_numpy())
print("list:", s.values.tolist())
```

work:

```
array: [11 22 33]
list: [11, 22, 33]
```



---

### 数据表DataFrame

Pandas 首先支持的是序列数据和表格数据，因为这两种是如常生活中最常用的数据保存和编辑格式了，你见过有人去编辑一个 3 维数据吗？

在上一节数据文件读取的教学中，你 load 到的数据，实际上就是一个 DataFrame， 举个最简单的例子。将一个二维数组变成 Pandas 的 DataFrame。

```python
df = pd.DataFrame([
  [1,2],
  [3,4]
])
df
```

work:

```
   0  1
0  1  2
1  3  4
```



___

## **选取数据**

Pandas 的数据选取，和 List，Numpy Array 还是有挺大差别的，因为它想要维护了很多的人类可读的索引信息， 所以它在索引的时候，也有不一样的处理方式.

面对应用比较多的工作学习场景，我先以 Excel 型的表格数据举例，我先构建一下下面这份 DataFrame：

```py
import pandas as pd
import numpy as np

data = np.arange(-12, 12).reshape((6, 4))
df = pd.DataFrame(
  data, 
  index=list("abcdef"), 
  columns=list("ABCD"))
df
```

work:

```
    A   B   C   D
a -12 -11 -10  -9
b  -8  -7  -6  -5
c  -4  -3  -2  -1
d   0   1   2   3
e   4   5   6   7
f   8   9  10  11
```

我想要选取第 2 到第 4 位数据的 A C 两个特征，这时咋办？ 想想 Pandas 这么牛逼，肯定有办法解决。的确，它解决的方法是采用索引转换的方式，比如我在 `.loc` 模式下，将序号索引转换成 `.loc` 的标签索引。

```python
row_labels = df.index[2:4]
print("row_labels:\n", row_labels)
print("\ndf:\n", df.loc[row_labels, ["A", "C"]])
```

work:

```
row_labels:
 Index(['c', 'd'], dtype='object')

df:
    A  C
c -4 -2
d  0  2
```

再看看 Column 的 labels 怎么取？

```python
col_labels = df.columns[[0, 3]]
print("col_labels:\n", col_labels)
print("\ndf:\n", df.loc[row_labels, col_labels])
```

work:

```
col_labels:
 Index(['A', 'D'], dtype='object')

df:
    A  D
c -4 -1
d  0  3
```

### 条件过滤筛选

按条件过滤其实是一件很有趣的事，因为很多情况我们事先也不知道具体的 index 是什么，我们更想要从某些条件中筛选数据。 下面我举几个例子，大家应该很容易 get 到其中的奥秘。

**选在 A Column 中小于 0 的那些数据**

```python
df[df["A"] < 0]
```



**选在第一行数据不小于 -10 的数据**，这里注意了你可以用两种方式，一种是 `~` 来表示 `非` 什么什么，第二种是直接用 `>=-10` 来筛选。

```python
print("~:\n", df.loc[:, ~(df.iloc[0] < -10)])
print("\n>=:\n", df.loc[:, df.iloc[0] >= -10])

```



同上面类似的，我还能用或 `|` 来表示 or 的意思, `&` 表述 and。比如**选在第一行数据不小于 -10 或小于 -11 的数据**

```python
i0 = df.iloc[0]
df.loc[:, ~(i0 < -10) | (i0 < -11)]

```



----

## 数据远算处理

###  筛选赋值运算

在之前**筛选数据**的教学中，我们能成功找出数据中的某个部分， 那么针对这个找出的部分，我们对它进行操作也是没问题的。比如下面我们先生成一组数据，然后在对这组数据进行筛选运算。

```python
import pandas as pd
import numpy as np

data = np.arange(-12, 12).reshape((6, 4))
df = pd.DataFrame(
  data, 
  index=list("abcdef"), 
  columns=list("ABCD"))
df

```

work:

```
    A   B   C   D
a -12 -11 -10  -9
b  -8  -7  -6  -5
c  -4  -3  -2  -1
d   0   1   2   3
e   4   5   6   7
f   8   9  10  11
```

可以看到数据生成后的样子，下面，我们在筛选出 `A` column 出来，对 `A` column 进行乘以 0 的运算。

```python
df["A"] *= 0
df

```

work:

```
   A   B   C   D
a  0 -11 -10  -9
b  0  -7  -6  -5
c  0  -3  -2  -1
d  0   1   2   3
e  0   5   6   7
f  0   9  10  11
```

你看看，新的 df 里面 `A` column 是不是都变成 0 了！

同样，**筛选数据**教学中，提到的 `iloc` `loc` 功能也是可以用来对某数据进行运算的。 以免你忘记，我简单回顾一下，`iloc` 找的是 index，`loc` 找的是标签。

```python
df.loc["a", "A"] = 100
df.iloc[1, 0] = 200
df
```

work:

```
     A   B   C   D
a  100 -11 -10  -9
b  200  -7  -6  -5
c    0  -3  -2  -1
d    0   1   2   3
e    0   5   6   7
f    0   9  10  11
```

看看现在 `a` row `A` column 的值是不是被改成了 100， 而第 [1, 0] 位是不是被改成了 200？这只是赋值，现在你拿这些赋值的方法进行运算试试？ 比如：

```
df.loc["a", :] = df.loc["a", :] * 2
df
```

workk:

```
     A   B   C   D
a  200 -22 -20 -18
b  200  -7  -6  -5
c    0  -3  -2  -1
d    0   1   2   3
e    0   5   6   7
f    0   9  10  11
```

我们再来试试按条件运算？下面做的是对于 `df["A"]`, 我要找出 `df["A"]` 中等于 0 的数，把这些数赋值成 -1。

```python
df["A"][df["A"] == 0] = -1
df
```

work:

```
     A   B   C   D
a  200 -22 -20 -18
b  200  -7  -6  -5
c   -1  -3  -2  -1
d   -1   1   2   3
e   -1   5   6   7
f   -1   9  10  11
```

基本上，pandas 中可以用于筛选数据的方法都可以用来进一步把筛选出来的数据赋予新的值。



---

### Apply方法

另一种比较方便的批处理数据的方法，我比较喜欢用的是 `apply`。这是一种可以针对数据做自定义功能的运算。意味着可以简化数据做复杂的功能运算。 上面我们提到的筛选运算，其实是一种简单的运算方式，如果当运算变得复杂，甚至还需要很多局部变量来缓存运算结果，我们就可以尝试把运算过程放置在一个 `func` 中， 模块化。

举个例子，我有下面这批数据

```python
df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
df
```

work:

```
   A  B
0  4  9
1  4  9
2  4  9
```

我要对这个 `df` 做全量的平方根计算。 用一般能想到的方式，就会是下面这样。

```python
np.sqrt(df)

```

work:

```
     A    B
0  2.0  3.0
1  2.0  3.0
2  2.0  3.0
```

如果用 `apply`，就会变成

```python
df.apply(np.sqrt)

```

work:

```
     A    B
0  2.0  3.0
1  2.0  3.0
2  2.0  3.0
```

我们把 `np.sqrt` 这个函数当成一个参数传入了 `apply`，看起来好像并没啥用，还不如直接写 `np.sqrt(df)` 来的方便。的确这个 case 写 `np.sqrt(df)` 是要简单点。 但是下面这种 case 呢？

```python
def func(x):
    return x[0] * 2, x[1] * -1

df.apply(func, axis=1, result_type='expand')

```

work:

```
   0  1
0  8 -9
1  8 -9
2  8 -9
```

这种极度自定义的功能，对 `df` 中的每一行，每行第 0 位乘以 2，第 1 位乘以 -1，我们原本的 col0，就都乘了 2，而 col1 就都乘了-1。 提示一下，`apply` 里面还有不同的参数项可以选，我使用了一个 `result_type="expand"` 的配置，让输出的结果可以生成多 column，要不然， 会只生成一个 column，所有的结果都写在这一个 column 里。要不你试试取消 `result_type`，观察一下生成结果的变化。

顺带提一下，如果 `reult_type="broadcast"`，那么原 column 和 index 名会继承到新生成的数据中。仔细对比上下两次的运行，你就能发现不同的表现了。

```python
def func(x):
    return x[0] * 2, x[1] * -1

df.apply(func, axis=1, result_type='broadcast')

```

work:

```
   A  B
0  8 -9
1  8 -9
2  8 -9
```

若你只想改一个 column

```python
def func(x):
    return x["A"] * 4

df.apply(func, axis=1)

```

work:

```
0    16
1    16
2    16
dtype: int64
```

若你还是返回原 df，但只有一个 column 被修改了？

```python
def func(x):
    return x["A"] * 4

df["A"] = df.apply(func, axis=1)
df

```

work:

```
    A  B
0  16  9
1  16  9
2  16  9
```

若你你还想只对 row 进行操作？调一下 `axis=0` 和 `func` 里的运算规则。

```py
def func(r):
    return r[2] * 4

last_row = df.apply(func, axis=0)
print("last_row:\n", last_row)

df.iloc[2, :] = last_row
print("\ndf:\n", df)

```

work:

```
last_row:
 A    64
B    36
dtype: int64

df:
     A   B
0  16   9
1  16   9
2  64  36
```



---

##  文字处理

相比 Python 的科学运算神器 **Numpy**，Pandas 还有一个特别优势的地方，那就是处理数据库当中的文字信息。 对比 Numpy，Numpy 是一个纯数据处理的库，在数据处理的速度上， 是要优于 Pandas 的。但是在处理数据的丰富度上，比如要处理文字，日期型数据的时候，Pandas 还是有很大优势的。

我这里要介绍以下功能：

- 格式化字符
  - `str.upper(); str.lower(); str.len()`
  - `str.strip(); str.lstrip(); str.rstrip()`
  - `str.split()`
- 正则方案
  - `str.contains(); str.match(); str.startswith(); str.endswith()`
  - `str.replace()`
  - `str.extract(); str.extractall()`
- 拼接
  - `str.cat()`

###  格式化字符

首先，我觉得我需要对标一下 Python 中**自带的文字处理功能**。 Python 本身就有很多自带的文字函数。 比如 `strip()` , `upper()` 等，我们就来对应着学习吧。

```python
import pandas as pd

py_s = "A,B,C,Aaba,Baca,CABA,dog,cat"
pd_s = pd.Series(
    ["A", "B", "C", "Aaba", "Baca", "CABA", "dog", "cat"],
    dtype="string")

print("python:\n", py_s.upper())
print("\npandas:\n", pd_s.str.upper())

```

work:

```
python:
 A,B,C,AABA,BACA,CABA,DOG,CAT

pandas:
 0       A
1       B
2       C
3    AABA
4    BACA
5    CABA
6     DOG
7     CAT
dtype: string
```



**注意如果要用到 Pandas 丰富的文字处理功能，你要确保 Series 或者 DataFrame 的 `dtype="string"`**，如果不是 string， 比如我们刚从一个 **excel 中读取出来**一个数据，自动读的，没有解析到 string 格式， 我们怎么调整呢？ 其实也简单。

```python
pd_not_s = pd.Series(
    ["A", "B", "C", "Aaba", "Baca", "CABA", "dog", "cat"],
)
print("pd_not_s type:", pd_not_s.dtype)
pd_s = pd_not_s.astype("string")
print("pd_s type:", pd_s.dtype)

```

work:

```
pd_not_s type: object
pd_s type: string
```



好，牢记这点，我们接着来对比原生 Python 的功能。

```python
print("python lower:\n", py_s.lower())
print("\npandas lower:\n", pd_s.str.lower())
print("python len:\n", [len(s) for s in py_s.split(",")])
print("\npandas len:\n", pd_s.str.len())

```

work:

```1
python lower:
 a,b,c,aaba,baca,caba,dog,cat

pandas lower:
 0       a
1       b
2       c
3    aaba
4    baca
5    caba
6     dog
7     cat
dtype: string
python len:
 [1, 1, 1, 4, 4, 4, 3, 3]

pandas len:
 0    1
1    1
2    1
3    4
4    4
5    4
6    3
7    3
dtype: Int64
```



再来对比一下对文字的裁剪：

```python
py_s = ["   jack", "jill ", "    jesse    ", "frank"]
pd_s = pd.Series(py_s, dtype="string")
print("python strip:\n", [s.strip() for s in py_s])
print("\npandas strip:\n", pd_s.str.strip())

print("\n\npython lstrip:\n", [s.lstrip() for s in py_s])
print("\npandas lstrip:\n", pd_s.str.lstrip())

print("\n\npython rstrip:\n", [s.rstrip() for s in py_s])
print("\npandas rstrip:\n", pd_s.str.rstrip())

```

work:

```
python strip:
 ['jack', 'jill', 'jesse', 'frank']

pandas strip:
 0     jack
1     jill
2    jesse
3    frank
dtype: string


python lstrip:
 ['jack', 'jill ', 'jesse    ', 'frank']

pandas lstrip:
 0         jack
1        jill 
2    jesse    
3        frank
dtype: string


python rstrip:
 ['   jack', 'jill', '    jesse', 'frank']

pandas rstrip:
 0         jack
1         jill
2        jesse
3        frank
dtype: string
```



从结果可能看不清空白符有多少，但是实际上是把空白符都移除掉了。 下面再对比一下 `split` 拆分方法。

```python
py_s = ["a_b_c", "jill_jesse", "frank"]
pd_s = pd.Series(py_s, dtype="string")
print("python split:\n", [s.split("_") for s in py_s])
print("\npandas split:\n", pd_s.str.split("_"))

```

work:

```
python split:
 [['a', 'b', 'c'], ['jill', 'jesse'], ['frank']]

pandas split:
 0        [a, b, c]
1    [jill, jesse]
2          [frank]
dtype: object
```



咦，pandas 这样拆分起来怪怪的，把结果都放到了一个 column 里面，我还记得用 `apply()` 的时候，我可以加一个 `result_type="expand"`，同样，在 `split` 中也有类似的功能，可以将拆分出来的结果放到不同的 column 中去。

```python
pd_s.str.split("_", expand=True)

```

work:

```
      0
0     A
1     B
2     C
3  Aaba
4  Baca
5  CABA
6   dog
7   cat
```



你看，一共拆出了三个 column，但是有些 column 因为没有 split 出那么多值，所以显示的也是 `pd.nan`

这里还有一点我想说，我们上面都是在 `Series` 里面做实验，其实 `DataFrame` 也是一样的。 **你要做的，只是先选一个 column 或者 row，拿到一个 Series 再开始做 str 的处理**

```python
pd_df = pd.DataFrame([["a", "b"], ["C", "D"]])
pd_df.iloc[0, :].str.upper()

```

work:

```
0    A
1    B
Name: 0, dtype: object
```



___

###  正则方案

正则是一个很有用的东西，我们在 **Python 基础**中也花了大功夫来学习正则表达式， 用特殊规则获取到特殊的文本。在演示的第一件事情就是它是否真的可以找到一些东西。我们用 `str.contains()` 或 `str.match()` 来确认它真的找到了匹配文字。

如果你还不了解正则表达式，我强烈建议你复习一下python基础。 要不然你也看不懂我写的匹配规则，比如这里 `[0-9][a-z]` 表示要匹配 0~9 的任何数字，之后再接着匹配 a~z 的任何字母。

```python
pattern = r"[0-9][a-z]"
s = pd.Series(["1", "1a", "11c", "abc"], dtype="string")
s.str.contains(pattern)

```

work:

```
0    False
1     True
2     True
3    False
dtype: boolean
```



现在请你把 `str.contains()` 换成 `str.match()` 看看结果有无变化。仔细的你肯定发现了，`11c` 这个字符，用 `contains()` 可以匹配， 但是 `match()` 却不能。那是因为 **只要包含正则规则，`contains` 就为 True， 但是 `match()` 的意思是你的正则规则要完全匹配才会返回 True。**

那么为了要让 `match` 匹配 `11c` 我们就需要把规则改成 `r"[0-9]+?[a-z]`。

```python
pattern = r"[0-9]+?[a-z]"
s.str.match(pattern)

```

work:

```\
0    False
1     True
2     True
3    False
dtype: boolean
```



还有一个十分有用，而且我觉得是最重要的，就是 `replace` 了，因为这真的减轻了我们很多复制粘贴的工作，比如 Excel 中人工按照一个规则修改老板给的新任务。 下面同样，我们对比 Python 原生的 replace，来验证一下。

```python
py_s = ["1", "1a", "21c", "abc"]
pd_s = pd.Series(py_s, dtype="string")
print("py_s replace '1' -> '9':\n", [s.replace("1", "9") for s in py_s])

print("\n\npd_s replace '1' -> '9':\n", pd_s.str.replace("1", "9"))

```

work:

```
py_s replace '1' -> '9':
 ['9', '9a', '29c', 'abc']


pd_s replace '1' -> '9':
 0      9
1     9a
2    29c
3    abc
dtype: string
```

但比原生 Python 强大的是，这个 replace 是支持正则的。我们把所有数字都替换成这个 `NUM` 吧。

```python
print("pd_s replace -> 'NUM':")
pd_s.str.replace(r"[0-9]", "NUM", regex=True)

```

work:

```
pd_s replace -> 'NUM':
0        NUM
1       NUMa
2    NUMNUMc
3        abc
dtype: string
```

除了替换原本文字里的东西，我们还可以去从原本文字里找到特定的文字。有点像正则中的 `findall` 函数。

```python
s = pd.Series(['a1', 'b2', 'c3'])
s.str.extract(r"([ab])(\d)")

```

work:

```
     0    1
0    a    1
1    b    2
2  NaN  NaN
```

`r"([ab])(\d)"` 这一个正则匹配我简单介绍一下，其中有两个括号，第一个括号是想提取的第一种规则，第二个是第二种想提取的规则。 那么运行出来，你会看到有两个 column，分别对应着这两个提取规则出来的值。最后一行出来的结果是两个 NaN，也就意味着第三个数据没有提取出来任何东西。

对应 `str.extract()` 还有一个 `str.extractall()` 函数，用来返回所有匹配，而不是第一次发现的匹配。
