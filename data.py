#data.py for gpu


import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torchvision.io import read_image
from PIL import Image
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_loaders(batch_size, classes=None):
    data_dir_train_real = '/content/quantum-vision-transformer-deepfake-detection/dataset_folder/real_vs_fake/real-vs-fake/train/real'
    data_dir_train_fake = '/content/quantum-vision-transformer-deepfake-detection/dataset_folder/real_vs_fake/real-vs-fake/train/fake'
    data_dir_test_real = '/content/quantum-vision-transformer-deepfake-detection/dataset_folder/real_vs_fake/real-vs-fake/test/real'
    data_dir_test_fake = '/content/quantum-vision-transformer-deepfake-detection/dataset_folder/real_vs_fake/real-vs-fake/test/fake'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalizing with a single channel mean/std
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalizing with a single channel mean/std
    ])

    def load_images_from_folder(folder, transform):
        images = []
        labels = []
        for filename in os.listdir(folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(folder, filename)
                image = read_image(img_path)
                image = transforms.ToPILImage()(image)  # Convert to PIL Image
                if transform:
                    image = transform(image)
                images.append(image.to(device))
                label = 0 if 'real' in folder else 1  # Assuming folder names indicate real/fake
                labels.append(label)
        return images, labels

    train_images_real, train_labels_real = load_images_from_folder(data_dir_train_real, train_transform)
    train_images_fake, train_labels_fake = load_images_from_folder(data_dir_train_fake, train_transform)
    test_images_real, test_labels_real = load_images_from_folder(data_dir_test_real, test_transform)
    test_images_fake, test_labels_fake = load_images_from_folder(data_dir_test_fake, test_transform)

    train_images = train_images_real + train_images_fake
    train_labels = train_labels_real + train_labels_fake
    test_images = test_images_real + test_images_fake
    test_labels = test_labels_real + test_labels_fake

    train_dataset = data.TensorDataset(torch.stack(train_images).to(device), torch.tensor(train_labels).to(device))
    test_dataset = data.TensorDataset(torch.stack(test_images).to(device), torch.tensor(test_labels).to(device))

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, test_loader
