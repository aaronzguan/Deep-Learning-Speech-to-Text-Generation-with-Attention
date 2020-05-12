import torch
from argument import test_args
from loader import get_loader
from model import create_model
from util import *
import pandas as pd
import os


if __name__ == '__main__':
    args = test_args()
    test_loader = get_loader(args, 'test', shuffle=False)

    model = create_model(args, isTest=True)
    model.load_model(args.model_path)
    model.eval()

    result = []
    test_num = 0
    with torch.no_grad():
        for i, (x_padded, x_lens) in enumerate(test_loader):
            x_padded = x_padded.to(args.device)
            predict_labels = model(x_padded, x_lens)
            predict_transcipt = ''
            for l in predict_labels:
                predict_transcipt += idx2chr[l]
            # predict_transcipt = greedy_decode(predict_labels)
            # print(predict_transcipt)
            result.append(predict_transcipt)
            test_num += len(x_lens)
            print(i, '/', len(test_loader))

    d = {'Id': list(range(test_num)), 'Predicted': result}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.split(args.model_path)[0] + '/' + args.result_file, header=True, index=False)
    print('Done')