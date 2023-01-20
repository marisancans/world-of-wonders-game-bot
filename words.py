import re

def load_words():
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words


def get_possible_matches(possible_words, word):
    possible_words = [x for x in possible_words if len(x) == word.letters]

    chars = [x.char for x in word.letters]

    r = re.compile(f'^[{"".join(chars)}]+$')
    newlist = list(filter(r.match, possible_words)) 

    print(newlist)
    return newlist

if __name__ == '__main__':
    english_words = load_words()
    # demo print
    print('fate' in english_words)