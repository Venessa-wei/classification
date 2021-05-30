import os
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def get_figure(x, y1, y2):

    # with legend
    f2 = plt.figure()
    p1 = plt.scatter(x, y1, marker='o', color='m', label='avg acc')
    p2 = plt.scatter(x, y2, marker='+', color='c', label='avg macro f1')
    plt.legend(loc='upper right')
    plt.show()


def main():
    accuracy = []
    f1 = []
    _dir = 'KNNlogs'
    ks = range(1, 50)
    for i in ks:
        filename = os.path.join(_dir, f'KNN-{i}.log')
        with open(filename) as fin:
            for line in fin.readlines():
                line = line.strip()
                # if line.find("accuracy") != -1:
                #     accuracy.append(line.split()[1])
                # elif line.find("macro avg") != -1:
                #     f1.append(line.split()[4])
                if line.find("Average Acc:") != -1:
                    accuracy.append(float(line.split()[2]))
                elif line.find("Average F1:") != -1:
                    f1.append(float(line.split()[2]))
    print(">>>>", accuracy)
    print(">>>>", f1)
    get_figure(ks, accuracy, f1)


if __name__ == '__main__':
    main()
