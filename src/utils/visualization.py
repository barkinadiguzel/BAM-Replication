import torch
import matplotlib.pyplot as plt


def visualize_channel_attention(attention):
    att = attention.squeeze().detach().cpu()
    att = att.mean(dim=0)

    plt.figure()
    plt.plot(att)
    plt.title("Channel Attention")
    plt.xlabel("Channel")
    plt.ylabel("Weight")
    plt.show()


def visualize_spatial_attention(attention):
    att = attention.squeeze().detach().cpu()

    plt.figure()
    plt.imshow(att, cmap="hot")
    plt.colorbar()
    plt.title("Spatial Attention")
    plt.show()
