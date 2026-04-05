"""
Miniproject 2 — Training entry point.

Loads the noisy image pairs from the course dataset, trains the from-scratch
framework defined in model.py (no torch.nn, no torch.autograd), saves the
weights as a text file, and prints PSNR on the validation set.

Dataset files (not included — obtained from the EPFL Deep Learning course):
    ../data/train_data.pkl  — tuple (noisy_imgs_1, noisy_imgs_2), each (N, C, H, W)
    ../data/val_data.pkl    — tuple (noisy_imgs, clean_imgs), each (N, C, H, W)
"""

import os
import torch
import matplotlib.pyplot as plt

from model import Model as md


def show(noisy_img, predict_img, clean_img):
    """Save a side-by-side comparison of noisy input, prediction, and clean target."""
    fig = plt.figure(figsize=(10, 7))
    rows, columns = 1, 3

    fig.add_subplot(rows, columns, 1)
    plt.imshow(noisy_img.permute(1, 2, 0))

    fig.add_subplot(rows, columns, 2)
    plt.imshow(predict_img.permute(1, 2, 0))

    fig.add_subplot(rows, columns, 3)
    plt.imshow(clean_img.permute(1, 2, 0))

    plt.savefig("image.png")


def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


if __name__ == '__main__':
    data_dir = os.path.join('..', 'data')
    noisy_imgs_1, noisy_imgs_2 = torch.load(os.path.join(data_dir, 'train_data.pkl'))
    noisy_imgs, clean_imgs = torch.load(os.path.join(data_dir, 'val_data.pkl'))

    model = md()

    # To continue from saved weights, uncomment:
    # model.load_pretrained_model('bestmodel.pth')

    model.train(noisy_imgs_1, noisy_imgs_2, 4, verbose=True)

    model.save('bestmodel.pth')

    predict_imgs = model.predict(noisy_imgs)

    print(compute_psnr(predict_imgs / 255.0, clean_imgs / 255.0))
    show(noisy_imgs[0], predict_imgs[0], clean_imgs[0])
