import numpy as np
import torchvision

from model import simple_gan
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
noise_dim = 50  # noise
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 200

# dataset
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
dataset = datasets.MNIST(root="data/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
fixed_noise = torch.randn((batch_size, noise_dim)).to(device)

D = simple_gan.Discriminator(image_dim).to(device)
G = simple_gan.Generator(noise_dim, image_dim).to(device)
opt_disc = torch.optim.Adam(D.parameters(), lr=lr)
opt_gen = torch.optim.Adam(G.parameters(), lr=lr)
criterion = nn.BCELoss()  # 二分类交叉熵损失函数

total_step = len(loader)
for epoch in range(num_epochs):
    # GAN不需要真实label
    for i, (images, labels) in enumerate(loader):
        images = images.view(-1, 784).to(device)
        batch_size = images.shape[0]
        # 训练判别器: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_img = G(noise)  # 根据随机噪声生成虚假数据
        disc_fake = D(fake_img)  # 判别器判断生成数据为真的概率
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # 虚假数据与0计算损失
        disc_real = D(images)  # 判别器判断真实数据为真的概率
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))  # 真实数据与1计算损失
        lossD = (lossD_real + lossD_fake) / 2
        D.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}],'
                  f' Step [{i + 1}/{total_step}],'
                  f' D_Real_Loss: {lossD_real.item():.4f},'
                  f' D_Fake_Loss: {lossD_fake.item():.4f},'
                  f' D_Loss: {lossD.item():.4f}')

        # 训练生成器: 在此过程中将判别器固定，min log(1 - D(G(z))) <-> max log(D(G(z))
        output = D(fake_img)
        lossG = criterion(output, torch.ones_like(output))
        G.zero_grad()
        lossG.backward()
        opt_gen.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}],'
                  f' Step [{i + 1}/{total_step}],'
                  f' G_Loss: {lossG.item():.4f}')

    with torch.no_grad():
        # 用固定的噪声数据生成图像，以对比经过不同epoch训练后的生成器的生成能力
        fake_img = G(fixed_noise).reshape(-1, 1, 28, 28)
        img_grid_fake = torchvision.utils.make_grid(fake_img, normalize=True)
        img_grid_fake_np = img_grid_fake.numpy()
        img_grid_fake_np = np.transpose(img_grid_fake_np, (1, 2, 0))
        img_grid_fake_np = (img_grid_fake_np * 255).astype(np.uint8)
        cv2.imwrite(f'./epoch_{epoch + 1}.jpg', img_grid_fake_np)
