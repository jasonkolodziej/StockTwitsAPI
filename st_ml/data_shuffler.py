import csv
import random
if __name__ == "__main__":
    # shuffle the training data

    with open('training_data/result.csv', 'r') as r, open('training_data/shuffled.csv', 'a+') as w:
        data = r.readlines()
        header, rows = data[0], data[1:]
        random.shuffle(rows)
        rows = '\n'.join([row.strip() for row in rows])
        w.write(header + rows)