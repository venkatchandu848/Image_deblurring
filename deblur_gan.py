import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# Generator Network
class WIREBlock(nn.Module):
    def __init__(self, channels, omega=20.0, scale=30.0):
        super(WIREBlock, self).__init__()
        self.omega_0 = omega
        self.scale_0 = scale
        
        self.freq_modulation = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.scale_modulation = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        omega = self.omega_0 * self.freq_modulation(x)
        scale = self.scale_modulation(x) * self.scale_0
        return torch.cos(omega) * torch.exp(-(scale**2))

class EnhancedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.wire = WIREBlock(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.wire(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Initial feature extraction
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # WIRE-enhanced residual blocks
        self.res_blocks = nn.Sequential(
            EnhancedResidualBlock(256),
            EnhancedResidualBlock(256),
            EnhancedResidualBlock(256),
            EnhancedResidualBlock(256),
            EnhancedResidualBlock(256),
            EnhancedResidualBlock(256)  # Increased number of blocks
        )
        
        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        init_features = self.init_conv(x)
        
        # Encoding
        e1 = self.enc1(init_features)
        e2 = self.enc2(e1)
        
        # WIRE-enhanced residual processing
        res = self.res_blocks(e2)
        
        # Decoding with skip connections
        d1 = self.dec1(res)
        d2 = self.dec2(d1)
        
        # Final reconstruction
        out = self.final(d2)
        
        return out

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),  # 128x128
            *discriminator_block(64, 128),                 # 64x64
            *discriminator_block(128, 256),                # 32x32
            *discriminator_block(256, 512),                # 16x16
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)  # 1x1
        )
        
    def forward(self, x):
        return self.model(x)

# Training Class
class DeblurGAN_training():
    def __init__(self, dataloader, num_epochs=100, lr=0.0002, b1=0.5, b2=0.999):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        
        # Initialize generator and discriminator
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.content_loss = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        # Create Gaussian kernel for blurring
        self.Gaussian_kernel = self.create_gaussian_kernel().to(self.device)
        
        # Create directory for samples and metrics
        os.makedirs("samples", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)
        
        # Lists to store metrics
        self.psnr_blurred = []
        self.psnr_deblurred = []
        self.g_losses = []
        self.d_losses = []
        
        # Modify loss weights
        self.content_weight = 100.0  # Increased weight for content loss
        self.adversarial_weight = 0.001  # Reduced weight for adversarial loss
        
    def create_gaussian_kernel(self, kernel_size=21, sigma=2, channels=3):
        a = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(a, a, indexing='ij')
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        kernel = kernel / torch.sum(kernel)
        gaussian_kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        return gaussian_kernel
    
    def create_blurry_noisy_image(self, input):
        y_blurry = F.conv2d(input, self.Gaussian_kernel, groups=3, padding=10)
        y_blurry_noisy = y_blurry + torch.randn_like(y_blurry) * 0.05
        return y_blurry_noisy
    
    def train(self):
        for epoch in range(self.num_epochs):
            epoch_psnr_blurred = []
            epoch_psnr_deblurred = []
            epoch_g_losses = []
            epoch_d_losses = []
            
            for i, imgs in enumerate(tqdm(self.dataloader)):
                # Configure input
                real_imgs = imgs.to(self.device)
                blurred_imgs = self.create_blurry_noisy_image(real_imgs)
                
                # Calculate PSNR for blurred images
                psnr_blur = PSNR(blurred_imgs, real_imgs)
                epoch_psnr_blurred.append(psnr_blur.item())
                
                # Ground truths for adversarial loss
                valid = torch.ones((imgs.size(0), 1, 13, 13), requires_grad=False).to(self.device)
                fake = torch.zeros((imgs.size(0), 1, 13, 13), requires_grad=False).to(self.device)
                
                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                
                # Generate deblurred image
                deblurred_imgs = self.generator(blurred_imgs)
                
                # Calculate PSNR for deblurred images
                psnr_deblur = PSNR(deblurred_imgs, real_imgs)
                epoch_psnr_deblurred.append(psnr_deblur.item())
                
                # Adversarial loss
                d_fake = self.discriminator(deblurred_imgs.detach())
                d_loss_fake = self.adversarial_loss(d_fake, fake)
                
                # Content loss
                g_loss_content = self.content_loss(deblurred_imgs, real_imgs)
                
                # Modified generator loss calculation
                g_loss = self.content_weight * g_loss_content + self.adversarial_weight * d_loss_fake
                epoch_g_losses.append(g_loss.item())
                
                g_loss.backward()
                self.optimizer_G.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                
                # Real loss
                d_real = self.discriminator(real_imgs)
                d_loss_real = self.adversarial_loss(d_real, valid)
                
                # Fake loss
                d_fake = self.discriminator(deblurred_imgs.detach())
                d_loss_fake = self.adversarial_loss(d_fake, fake)
                
                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) / 2
                epoch_d_losses.append(d_loss.item())
                
                d_loss.backward()
                self.optimizer_D.step()
                
                if i % 100 == 0:
                    print(
                        f"[Epoch {epoch}/{self.num_epochs}] "
                        f"[Batch {i}/{len(self.dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] "
                        f"[G loss: {g_loss.item():.4f}] "
                        f"[PSNR Blurred: {psnr_blur.item():.2f}] "
                        f"[PSNR Deblurred: {psnr_deblur.item():.2f}]"
                    )
                    
                    # Save sample images
                    if i % 500 == 0:
                        self.save_sample_images(epoch, i, real_imgs, blurred_imgs, deblurred_imgs)
            
            # Calculate and store average metrics for the epoch
            avg_psnr_blurred = sum(epoch_psnr_blurred) / len(epoch_psnr_blurred)
            avg_psnr_deblurred = sum(epoch_psnr_deblurred) / len(epoch_psnr_deblurred)
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            
            self.psnr_blurred.append(avg_psnr_blurred)
            self.psnr_deblurred.append(avg_psnr_deblurred)
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            # Save metrics plot
            self.plot_metrics(epoch)
    
    def plot_metrics(self, epoch):
        plt.figure(figsize=(15, 10))
        
        # Plot PSNR values
        plt.subplot(2, 1, 1)
        plt.plot(self.psnr_blurred, label='Blurred', color='red', linestyle='--')
        plt.plot(self.psnr_deblurred, label='Deblurred', color='green')
        plt.axhline(y=30, color='blue', linestyle=':', label='Good Quality Threshold')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('PSNR Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot PSNR improvement
        plt.subplot(2, 1, 2)
        improvements = [d - b for d, b in zip(self.psnr_deblurred, self.psnr_blurred)]
        plt.plot(improvements, label='PSNR Improvement', color='purple')
        plt.axhline(y=2, color='red', linestyle=':', label='Minimum Target Improvement')
        plt.axhline(y=5, color='green', linestyle=':', label='Good Improvement')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR Improvement (dB)')
        plt.title('Deblurring Quality Improvement')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"metrics/metrics_epoch_{epoch}.png")
        plt.close()
        
        # Save metrics to text file with quality assessment
        with open(f"metrics/metrics_epoch_{epoch}.txt", "w") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"Average PSNR Blurred: {self.psnr_blurred[-1]:.2f} dB\n")
            f.write(f"Average PSNR Deblurred: {self.psnr_deblurred[-1]:.2f} dB\n")
            f.write(f"PSNR Improvement: {improvements[-1]:.2f} dB\n")
            
            # Quality assessment
            f.write("\nQuality Assessment:\n")
            if self.psnr_deblurred[-1] > 40:
                f.write("Deblurred Image Quality: Excellent\n")
            elif self.psnr_deblurred[-1] > 30:
                f.write("Deblurred Image Quality: Good\n")
            elif self.psnr_deblurred[-1] > 20:
                f.write("Deblurred Image Quality: Acceptable\n")
            else:
                f.write("Deblurred Image Quality: Poor\n")
                
            if improvements[-1] > 5:
                f.write("Deblurring Performance: Excellent (>5dB improvement)\n")
            elif improvements[-1] > 2:
                f.write("Deblurring Performance: Good (>2dB improvement)\n")
            else:
                f.write("Deblurring Performance: Needs Improvement\n")
            
            f.write(f"\nAverage Generator Loss: {self.g_losses[-1]:.4f}\n")
            f.write(f"Average Discriminator Loss: {self.d_losses[-1]:.4f}\n")
    
    def save_sample_images(self, epoch, batch_i, real_imgs, blurred_imgs, deblurred_imgs):
        # Denormalize images
        real_imgs = real_imgs * 0.5 + 0.5
        blurred_imgs = blurred_imgs * 0.5 + 0.5
        deblurred_imgs = deblurred_imgs * 0.5 + 0.5
        
        # Create figure
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        for i in range(3):
            # Plot real images
            axs[i, 0].imshow(real_imgs[i].permute(1, 2, 0).cpu().detach().numpy())
            axs[i, 0].set_title("Original")
            axs[i, 0].axis("off")
            
            # Plot blurred images
            axs[i, 1].imshow(blurred_imgs[i].permute(1, 2, 0).cpu().detach().numpy())
            axs[i, 1].set_title("Blurred")
            axs[i, 1].axis("off")
            
            # Plot deblurred images
            axs[i, 2].imshow(deblurred_imgs[i].permute(1, 2, 0).cpu().detach().numpy())
            axs[i, 2].set_title("Deblurred")
            axs[i, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"samples/epoch_{epoch}_batch_{batch_i}.png")
        plt.close()

# PSNR calculation function
def PSNR(x, gt):
    return 10 * torch.log10(torch.max(x) / torch.mean((x - gt) ** 2)) 