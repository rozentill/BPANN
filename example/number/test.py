
import numpy as np
import struct
import matplotlib.pyplot as plt



filename = './MINIST/train-labels-idx1-ubyte'
binfile = open(filename , 'rb')
buf = binfile.read()

index = 0
magic, numItem = struct.unpack_from('>II' , buf , index)
index += struct.calcsize('>II')

im = struct.unpack_from('>1B' ,buf, index)
index += struct.calcsize('>1B')
im = struct.unpack_from('>1B' ,buf, index)
index += struct.calcsize('>1B')
im = struct.unpack_from('>1B' ,buf, index)
index += struct.calcsize('>1B')
im = struct.unpack_from('>784B' ,buf, index)
index += struct.calcsize('>784B')


# pm = [0 for x in range(0,784)]
#
# for i in range(0,784):
#     if im[i]>100:
#         pm[i]=255
#     else:
#         pm[i]=0
#
# pm = np.array(pm)
# pm = pm.reshape(28,28)
# im /= 256.0
# print im/256.0
print im[0]
print numItem

# fig = plt.figure()
# plotwindow = fig.add_subplot(111)
# plt.imshow(pm)
# plt.show()
