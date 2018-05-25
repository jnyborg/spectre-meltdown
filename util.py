def get_accuracy(test, actual):
    correct = sum(1 for a, b in zip(test, actual) if a == b)
    return float(correct) / max(len(test), len(actual))

def majority_vote(strings):
    merged = ""
    temp = {}
    char_index = 0

    working_strings = []

    for i in range(len(strings)):
        working_strings.insert(i, { 'arr': strings[i], 'length': len(strings[i]), 'active': True })

    while any([w['active'] for w in working_strings]):
        for current_array in working_strings:
            if not current_array['active']: # File is no longer active
                continue
            
            if current_array['arr'][char_index] in temp.keys():
                temp[current_array['arr'][char_index]] += 1 # Increment the vote of this char
            else:
                temp[current_array['arr'][char_index]] = 1
            
            if char_index + 1 >= len(current_array['arr']): # File is empty now
                current_array['active'] = False
        
        merged += max(temp, key=temp.get)
        if len(temp) > 1:
            max_occur = max(temp, key=temp.get)
            #print(temp)
            #print("{} out of {} agreed on {}".format(temp[max_occur], len([w for w in working_strings if w['active']]), max_occur))
        char_index += 1
        temp = {}
    
    return merged