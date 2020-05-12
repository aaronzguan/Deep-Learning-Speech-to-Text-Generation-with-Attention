import os
import numpy as np
import torch
from argument import train_args
from loader import get_loader
from model import create_model
from tqdm import tqdm


def save_log(message, args):
    log_name = os.path.join(args.expr_dir, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('\n' + message)


def validate(args, model, val_loader):
    model.eval()
    with torch.no_grad():
        running_batch_num, running_perplexity, running_dist = 0, 0.0, 0.0
        for i, (x_padded, x_lens, y_padded, y_lens) in enumerate(val_loader):
            x_padded, y_padded = x_padded.to(args.device), y_padded.to(args.device)
            predict_labels, perplexity, edit_distance = model(x_padded, x_lens, y_padded, y_lens, validation=True)
            running_perplexity += perplexity
            running_dist += edit_distance
            running_batch_num += 1

    return running_perplexity/running_batch_num, running_dist/running_batch_num


if __name__ == '__main__':
    args = train_args()

    from shutil import copyfile
    copyfile('model.py', os.path.join(args.expr_dir, 'model.py'))

    train_loader = get_loader(args, name='train', shuffle=True)
    val_loader = get_loader(args, name='val', shuffle=False)

    model = create_model(args, isTest=False)
    if args.fine_tune:
        model.load_model(args.fine_tune_model_path)
    model.train_setup()
    cur_dist = np.inf
    pbar = tqdm(range(1, args.num_epochs + 1), ncols=0)
    for epoch in pbar:
        model.train()

        running_batch, running_perplexity, running_dist = 0, 0.0, 0.0
        for i, (x_padded, x_lens, y_padded, y_lens) in enumerate(train_loader):
            x_padded, y_padded = x_padded.to(args.device), y_padded.to(args.device)
            predict_labels, perplexity, edit_distance = model(x_padded, x_lens, y_padded, y_lens, validation=False)
            running_perplexity += perplexity
            running_dist += edit_distance
            running_batch += 1
            model.optimize_parameters()
            lr = model.get_learning_rate()

            if (i + 1) % args.check_step == 0:  # print every check_step mini-batches
                message = '[%d, %5d] lr: %.5f perplexity: %.5f edit_distance: %.5f' % \
                          (epoch, i + 1, lr, running_perplexity/running_batch, running_dist/running_batch)
                pbar.set_description(desc=message)
                save_log(message, args)
                predict_transcipt, target_trascript = model.get_transcipt()

        # Update learning
        model.update_learning_rate(running_perplexity/running_batch)

        """
        Validation
        """
        if epoch % args.eval_step == 0:
            val_perplexity, val_dist = validate(args, model, val_loader)
            message = 'Epoch: %d, val perplexity %.5f, val edit-distance: %.5f' % (epoch, val_perplexity, val_dist)
            pbar.set_description(desc=message)
            save_log(message, args)
            if val_dist < cur_dist:
                model.save_model(epoch)
                cur_dist = val_dist
                message = 'Model saved at {} epoch !'.format(epoch)
                pbar.set_description(desc=message)
                save_log(message, args)
            message = '-' * 20
            save_log(message, args)












