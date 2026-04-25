# 多层的感知机实现
# 实现 异或门

# 与门
def AND(x1,x2):
    w1 = 0.5
    w2 = 0.5
    xita = 0.7
    return 1 if x1*w1 + x2*w2 > xita else 0

# 与非门
def NAND(x1,x2):
    w1 = -0.5
    w2 = -0.5
    xita = -0.7
    return 1 if x1*w1 + x2*w2 > xita else 0

# 或门
def OR(x1,x2):
    w1 = 0.5
    w2 = 0.5
    xita = 0
    return 1 if x1*w1 + x2*w2 > xita else 0

# 多层感知机实现
# 异或门
def XOR(x1,x2):
    return AND(NAND(x1,x2),OR(x1,x2))



if __name__ == '__main__':
    print("="*10,"与门","="*10)
    print(AND(1,1))
    print(AND(0,1))
    print(AND(1,0))
    print(AND(0,0))
    print("="*10,"与非门","="*10)
    print(NAND(1,1))
    print(NAND(0,1))
    print(NAND(1,0))
    print(NAND(0,0))
    print("="*10,"或门","="*10)
    print(OR(1,1))
    print(OR(0,1))
    print(OR(1,0))
    print(OR(0,0))
    print("="*10,"异或门","="*10)
    print(XOR(1,1))
    print(XOR(0,1))
    print(XOR(1,0))
    print(XOR(0,0))
