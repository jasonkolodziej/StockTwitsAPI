import csv
import random
if __name__ == "__main__":
    # shuffle the training data

    with open('training_data/full_shuffled_ts.csv', 'r') as r, open('training_data/train.csv', 'a+') as tr, open('training_data/valid.csv', 'a+') as v, open('training_data/test.csv', 'a+') as te:
        all_data = r.readlines()
        all_data_c = len(all_data)
        train = all_data[:int(0.8*all_data_c)]
        test = all_data[int(0.8*all_data_c):int(0.9*all_data_c)]
        valid = all_data[int(0.9*all_data_c):]
        tr.writelines(train)
        v.writelines(valid)
        te.writelines(test)