import os
import json

if __name__ == "__main__":
    with open('samples/dataSample.json') as json_file:
        data = json.load(json_file)
        i = 0
        for p in data['messages']:
            i += 1
        print('posts count', i)