from deblur_gan import DeblurGAN_training
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def main():
    # Hyperparameters
    img_dir = 'D:\Projects\Image_deblurring\images'  # Replace with your dataset path
    batch_size = 4
    num_epochs = 200
    lr = 0.0002
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset and dataloader
    dataset = ImageDataset(img_dir, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize and train the GAN
    gan_trainer = DeblurGAN_training(
        dataloader=dataloader,
        num_epochs=num_epochs,
        lr=lr
    )
    
    # Start training
    gan_trainer.train()

if __name__ == "__main__":
    main() 