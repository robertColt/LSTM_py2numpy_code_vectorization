# import numpy as np  #
#
#
# next= None  #
# translation = None  #
# a = np.array([2])  #
# b = np.array([3])  #
# c = np.array([4])  #
# n = 1  #
# m = 1  #
# o = 1  #

next
for i in range(n):
    c[i] = a[i] * b[i]
translation
c = np.dot(a, b)

next
for i in range(n):
    c[i] = b[i] * a[i]
translation
c = np.dot(b, a)

next
for i in range(len(c)):
    c[i] = a[i] * b[i]
translation
c = np.dot(a, b)

next
for i in range(len(c)):
    c[i] = b[i] * a[i]
translation
c = np.dot(b, a)

next
for i in range(n):
    a[i] = c[i] * b[i]
translation
a = np.dot(c, b)

next
for i in range(n):
    a[i] = b[i] * c[i]
translation
a = np.dot(b, c)

next
for i in range(len(a)):
    a[i] = c[i] * b[i]
translation
a = np.dot(c, b)

next
for i in range(len(a)):
    a[i] = b[i] * c[i]
translation
a = np.dot(b, c)

next
for i in range(n):
    b[i] = a[i] * c[i]
translation
b = np.dot(a, c)

next
for i in range(n):
    b[i] = c[i] * a[i]
translation
b = np.dot(c, a)

next
for i in range(len(b)):
    b[i] = a[i] * c[i]
translation
b = np.dot(a, c)

next
for i in range(len(b)):
    b[i] = c[i] * a[i]
translation
b = np.dot(c, a)

next
for i in range(n):
    b = max(a[i], b)
translation
b = np.max(a)

next
for i in range(n):
    a = max(b[i], a)
translation
a = np.max(b)

next
for i in range(n):
    for j in range(m):
        b = max(a[i][j], b)
translation
b = np.max(a)

next
for i in range(n):
    for j in range(m):
        a = max(b[i][j], a)
translation
a = np.max(b)

next
for i in range(n):
    for j in range(m):
        b += a[i][j]
b = b / (n*m)
translation
b = np.average(a)

next
for i in range(n):
    for j in range(m):
        a += b[i][j]
a = a / (n*m)
translation
a = np.average(b)

next
for i in range(n):
    a += b[i]
a = a / n
translation
a = np.average(b)

next
for i in range(n):
    b += a[i]
b = b / n
translation
b = np.average(a)