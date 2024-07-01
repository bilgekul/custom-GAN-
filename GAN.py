
class GAN(nn.Module):
    def __init__(self, generator, discriminator, train_loader, num_epochs, input_dim, g_learning_rate, d_learning_rate, patience):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.num_epochs = num_epochs
        self.input_dim = input_dim
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate
        self.patience = patience
        self.early_stopping = EarlyStopping(patience=patience)
        self.train_loader = train_loader

    def noiser(self, batch_size, dim, device, mu=0, sigma=1):
        noise = torch.randn(batch_size, dim, device=device) * sigma + mu
        return noise

    def train_gan(self):
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate, betas=(0.5, 0.999))

        for epoch in range(self.num_epochs):
            g_losses = []
            d_losses = []
            for i, (images, _) in enumerate(self.train_loader):
                images = images.to(device)
                batch_size = images.size(0)
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = -torch.ones(batch_size, 1, device=device)

                optimizer_G.zero_grad()
                generated_images = self.generator(self.noiser(batch_size=batch_size, dim=self.input_dim, device=device))
                g_loss = -torch.mean(self.discriminator(generated_images))
                g_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                real_loss = torch.mean(F.relu(1 - self.discriminator(images)))
                fake_loss = torch.mean(F.relu(1 + self.discriminator(generated_images.detach())))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            g_loss_mean = sum(g_losses) / len(g_losses)
            d_loss_mean = sum(d_losses) / len(d_losses)

            print(f"Epoch [{epoch}/{self.num_epochs}], d_loss_mean: {d_loss_mean}, g_loss_mean: {g_loss_mean}")

            self.early_stopping(d_loss_mean)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(vutils.make_grid(generated_images.to("cpu"), padding=2, normalize=True).cpu(), (1, 2, 0)))
            plt.show()
