import torch
def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    # 防止Nan和Inf造成的错误
    if min_val==max_val and min_val==0:
        scale =(max_val - min_val)+0.001 / (qmax - qmin)
    elif min_val==max_val and min_val!=0:
        scale=max_val/(qmax-qmin)
    else:
        scale =(max_val - min_val) / (qmax - qmin)

    initial_zero_point = (qmin - min_val) / scale

    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)


    return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    return q_x.int(),scale,zero_point


def dequantize_tensor(scale,x,zero_point):
    scale=float(scale)
    return scale * (x.float() - zero_point)
