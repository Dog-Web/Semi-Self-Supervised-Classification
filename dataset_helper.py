import torch.utils
from torchvision import transforms
from PIL import Image
import random
import torch
from torchvision.datasets import ImageFolder

class JigsawPuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, tile_size=75, num_tiles_per_row=3, transform=None):
        self.tile_size = tile_size
        self.num_tiles_per_row = num_tiles_per_row
        self.image_size = tile_size * num_tiles_per_row
        self.data = ImageFolder(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resizing here to ensure uniformity
            transforms.ToTensor()
        ]) if transform is None else transform

    def __getitem__(self, index):
        img, _ = self.data[index]  # Get the original image
        img = img.resize((self.image_size, self.image_size))  # Resize image to ensure it's the correct size
        tiles = [img.crop((x, y, x + self.tile_size, y + self.tile_size))
                 for x in range(0, self.image_size, self.tile_size)
                 for y in range(0, self.image_size, self.tile_size)]
        random.shuffle(tiles)
        new_img = Image.new('RGB', (self.image_size, self.image_size))
        for i, tile in enumerate(tiles):
            x = (i % self.num_tiles_per_row) * self.tile_size
            y = (i // self.num_tiles_per_row) * self.tile_size
            new_img.paste(tile, (x, y))
        
        if self.transform:
            new_img = self.transform(new_img)  # Transform the jigsaw puzzle

        return new_img

    def __len__(self):
        return len(self.data)



class RotateImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.data = ImageFolder(root_dir)  # Using ImageFolder to handle loading
        # Add resizing to the transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize all images to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing with ImageNet standards
        ]) if transform is None else transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]  # Load an image and its label (label is ignored)
        
        # Randomly choose a rotation angle
        angle = random.choice([0, 90, 180, 270])
        rotated_img = img.rotate(angle)  # Rotate the image
        
        if self.transform:
            rotated_img = self.transform(rotated_img)  # Apply transformation
        
        # Convert angle to a categorical label: 0, 1, 2, or 3
        label = [0, 90, 180, 270].index(angle)
        
        return rotated_img, label