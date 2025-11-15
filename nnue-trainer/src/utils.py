from functools import lru_cache

import chess
import numpy as np


@lru_cache
def ctzll(x):
    return (x & -x).bit_length() - 1

def unpack_bits(packed_data, total_bits):
    if isinstance(packed_data, (bytes, bytearray)):
        arr = np.frombuffer(packed_data, dtype=np.uint8)
    else:
        arr = np.array(packed_data, dtype=np.uint8)

    unpacked = np.unpackbits(arr)

    unpacked = unpacked[:total_bits]
    return unpacked

    # return unpacked.tolist()


def pack_bits(data):
    packed_array = np.packbits(data)
    return bytearray(packed_array)