import numpy as np

c16 = np.complex128()
f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

c8 = np.complex64()
f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ = np.bool_()

b = bool()
c = complex()
f = float()
i = int()

AR = np.array([0], dtype=np.int64)
AR.setflags(write=False)

SEQ = (0, 1, 2, 3, 4)

# Time structures

dt > dt

td > td
td > i
td > i4
td > i8
td > AR
td > SEQ

# boolean

b_ > b
b_ > b_
b_ > i
b_ > i8
b_ > i4
b_ > u8
b_ > u4
b_ > f
b_ > f8
b_ > f4
b_ > c
b_ > c16
b_ > c8
b_ > AR
b_ > SEQ

# Complex

c16 > c16
c16 > f8
c16 > i8
c16 > c8
c16 > f4
c16 > i4
c16 > b_
c16 > b
c16 > c
c16 > f
c16 > i
c16 > AR
c16 > SEQ

c16 > c16
f8 > c16
i8 > c16
c8 > c16
f4 > c16
i4 > c16
b_ > c16
b > c16
c > c16
f > c16
i > c16
AR > c16
SEQ > c16

c8 > c16
c8 > f8
c8 > i8
c8 > c8
c8 > f4
c8 > i4
c8 > b_
c8 > b
c8 > c
c8 > f
c8 > i
c8 > AR
c8 > SEQ

c16 > c8
f8 > c8
i8 > c8
c8 > c8
f4 > c8
i4 > c8
b_ > c8
b > c8
c > c8
f > c8
i > c8
AR > c8
SEQ > c8

# Float

f8 > f8
f8 > i8
f8 > f4
f8 > i4
f8 > b_
f8 > b
f8 > c
f8 > f
f8 > i
f8 > AR
f8 > SEQ

f8 > f8
i8 > f8
f4 > f8
i4 > f8
b_ > f8
b > f8
c > f8
f > f8
i > f8
AR > f8
SEQ > f8

f4 > f8
f4 > i8
f4 > f4
f4 > i4
f4 > b_
f4 > b
f4 > c
f4 > f
f4 > i
f4 > AR
f4 > SEQ

f8 > f4
i8 > f4
f4 > f4
i4 > f4
b_ > f4
b > f4
c > f4
f > f4
i > f4
AR > f4
SEQ > f4

# Int

i8 > i8
i8 > u8
i8 > i4
i8 > u4
i8 > b_
i8 > b
i8 > c
i8 > f
i8 > i
i8 > AR
i8 > SEQ

u8 > u8
u8 > i4
u8 > u4
u8 > b_
u8 > b
u8 > c
u8 > f
u8 > i
u8 > AR
u8 > SEQ

i8 > i8
u8 > i8
i4 > i8
u4 > i8
b_ > i8
b > i8
c > i8
f > i8
i > i8
AR > i8
SEQ > i8

u8 > u8
i4 > u8
u4 > u8
b_ > u8
b > u8
c > u8
f > u8
i > u8
AR > u8
SEQ > u8

i4 > i8
i4 > i4
i4 > i
i4 > b_
i4 > b
i4 > AR
i4 > SEQ

u4 > i8
u4 > i4
u4 > u8
u4 > u4
u4 > i
u4 > b_
u4 > b
u4 > AR
u4 > SEQ

i8 > i4
i4 > i4
i > i4
b_ > i4
b > i4
AR > i4
SEQ > i4

i8 > u4
i4 > u4
u8 > u4
u4 > u4
b_ > u4
b > u4
i > u4
AR > u4
SEQ > u4
