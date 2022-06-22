import torch
from glob import glob
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import argparse

def main(model_path, np_data_path, result_path):
    model = torch.load(model_path)
    if os.path.isfile(np_data_path) and np_data_path.endswith("npz"):
        test_dataset = [np_data_path]
    elif os.path.isdir(np_data_path):
        test_dataset = sorted(glob(os.path.join(np_data_path, "*.npz")))
    else: 
        raise AssertionError("해당 경로가 npz 파일 혹은 해당 폴더 내 npz 파일이 있는 지 확인 하세요")

    for (i,filename) in enumerate(test_dataset):
        x = np.load(filename)
        X = x['x']

        X = X.reshape((-1, 1, 3000))
        Y = x['y']
        X = torch.tensor(X)
        X = X.cuda()
        predicted = model(X)
        predicted = predicted.cpu()
        predicted_class = np.argmax(predicted.detach().numpy(), axis = 1)
        if i == 0:
            preds = predicted_class
            gt = Y
            csv_path = os.path.join(result_path, f"result{i}.csv")
            with open(csv_path, "w") as f:
                lines = [str(_value) for _value in predicted_class]
                writer = csv.writer(f)
                writer.writerows(lines)

        else:
            preds = np.hstack([preds, predicted_class])
            csv_path = os.path.join(result_path, f"result{i}.csv")
            with open(csv_path, "w") as f:
                lines = [str(_value) for _value in predicted_class]
                writer = csv.writer(f)
                writer.writerows(lines)
            gt = np.hstack([gt, Y])

    distribution = confusion_matrix(gt, preds)
    plt.figure()
    ax = sns.heatmap(distribution, annot=True,fmt='g')
    plt.title('Confusion_matrix')
    plt.savefig(os.path.join(result_path, 'Confusion_matrix.png'))
    print(distribution)
    print(classification_report(gt, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-mp', '--model_path', type=str, default= "/home/jongwook95.lee/sleep_wave/attnsleep_new/saved/Exp1/17_06_2022_13_38_49_fold3/model_best.pt")
    parser.add_argument('-da', '--np_data_dir', type=str, default = "./numpy_data/test",
                        help='Directory containing numpy files')
    parser.add_argument('-rp', '--result_path', type=str, default="./")

    args = parser.parse_args()
    main(args.model_path, args.np_data_dir, args.result_path)
