import re
import helper
from pathlib import Path
from collections import Counter

def get_possible_matches(possible_words, aviable_chars, pattern):
    possible_words = [x for x in possible_words if len(x) == len(pattern)]

    possible_filtered = []

    for word in possible_words:
        f = []

        for char, p in zip(word, pattern):
            if p == "*":
                f.append("*")
                continue
            else:
                f.append(char)

        if f == pattern:
            possible_filtered.append(word)
            

    r = re.compile(f'^[{"".join(aviable_chars)}]+$')
    newlist = list(filter(r.match, possible_filtered)) 

    # Does letter count match
    count_filter = []
    pattern_count = dict(Counter(aviable_chars))

    for w in newlist:
        count = dict(Counter(w))
        f = []
        
        for k, v in count.items():
            f.append(v <= pattern_count[k])

        if all(f):
            count_filter.append(w)               

    return count_filter

def main():
    possible_words = helper.fs_json_load(Path("words_dictionary.json"))
    possible_words = list(possible_words.keys())
    # chars = [x.char for x in word.letters]

    chars = ["t", "f", "o", "n", "e"]

    word = ["*", "*", "*", "*", "n"]

    matches = get_possible_matches(possible_words, chars, word)
    print(matches)

if __name__ == '__main__':
    main()
