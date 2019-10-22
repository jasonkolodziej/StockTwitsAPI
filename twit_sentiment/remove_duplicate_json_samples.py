import json # import json library
import sys # import sys library

if __name__ == "__main__":
    """remove Duplicate JSON"""
    with open('samples/dataSample_merge_minify.json') as json_file:
        te = json.load(json_file)
        te = te['messages']
        print('original size was', len(te))
        unique = { repr(each): each for each in te }.values()
        # unique = { each['messages']['id'] : each for each in te }.values()
        print('new array count was', len(unique))
        f = open("samples/data_merge_rm_dup.json", "a+")
        unique = list(unique)
        s = json.dumps(unique)
        f.write(s)