import numpy as np
import os

path = 'metrics/unet/1-1-1'
files = os.listdir(path)
files.sort()


for file in files:
    data = np.load(os.path.join(path, file))
    if file == "acc.npy":
        print("Accuracy")
        for d in data:
            print(np.mean(d, axis=1))
        print('-' * 30)
    elif file == "dice.npy":
        print("Dice coefficient")
        print(np.max(np.mean(data, axis=2)))
        for d in data:
            print(np.mean(d, axis=1))
        print('-' * 30)
    elif file == "f1.npy":
        print("F1 score")
        for d in data:
            print(np.mean(d, axis=1))
        print('-' * 30)
    elif file == "hd.npy":
        print("Hausdorff distance")
        for d in data:
            n = []
            m = []
            for dd in d:
                m.append(np.mean(dd[dd != -1]))
                n.append(np.sum(dd == -1))
            print(m)
            print(n)
        print('-' * 30)
    elif file == "iou.npy":
        print("IOU")
        for d in data:
            f = []
            for j, dd in enumerate(d):
                f.append(np.sum(dd < 0.2))
            print(np.mean(d, axis=1))
            print(f)
        print('-' * 30)
    elif file == "precision.npy":
        print("Precision")
        for d in data:
            print(np.mean(d, axis=1))
        print('-' * 30)
    elif file == "recall.npy":
        print("Recall")
        for d in data:
            print(np.mean(d, axis=1))
        print('-' * 30)
    else:
        print(f"Cannot parse file: {file}")
        print('-' * 30)
