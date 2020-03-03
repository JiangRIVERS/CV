"""
Filename: GAN_test.py

A test for GAN according to https://github.com/RedstoneWill/MachineLearningInAction/blob/master/GAN/GAN_1.py

@author: Jiang Rivers
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def GAN_test():
    # A test for GAN
    plt.figure(figsize=(8, 6), dpi=80)
    plt.ion()

    batch = 64
    lr_D = 1e-4
    lr_G = 1e-4
    input_f = 5
    point = 15
    label_x = np.vstack([np.linspace(-1, 1, point) for i in range(batch)])

    label_y = np.sin(label_x * np.pi)
    label_y_tensor = torch.from_numpy(label_y).float()

    # Generative model
    G = nn.Sequential(
        nn.Linear(input_f, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, point)
    )

    # Discriminative model
    D = nn.Sequential(
        nn.Linear(point, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

    # optimizer
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_D)

    for step in range(2000):
        G_input = torch.randn(batch, input_f)
        fake = G(G_input)

        Dx = D(label_y_tensor)
        DGzi = D(fake)

        D_loss = -torch.mean(torch.log(Dx) + torch.log(1 - DGzi))
        G_loss = -torch.mean(torch.log(DGzi))

        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        # ploting
        if step % 20 == 0:
            plt.cla()
            plt.plot(label_x[0], fake[0].data.numpy(), c='#4AD631', lw=3, label='Generated painting')
            plt.plot(label_x[0], label_y[0], c='#74BCFF', lw=3, label='real painting')

            plt.text(-1, 0.75, 'D accurancy={:2f}'.format(Dx.data.numpy().mean()), fontdict={'size': 13})
            plt.text(-1, -0.5, 'epoch:{}'.format(step), fontdict={'size': 13})

            plt.ylim((-1, 1));
            plt.legend(loc='upper right', fontsize=10);
            plt.pause(0.05)

    plt.ioff()
    plt.show()

if __name__=='__main__':
    GAN_test()



