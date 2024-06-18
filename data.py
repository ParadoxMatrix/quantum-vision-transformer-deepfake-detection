import torch.utils.data as data
from torchvision import transforms
from torchvision.io import read_video
import os

def get_loaders(batch_size, classes=None):
    data_dir_real = 'dataset_folder/Celeb-real'
    data_dir_fake = 'dataset_folder/Celeb-synthesis'

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

    def load_videos_from_folder(folder, transform):
        videos = []
        labels = []
        for filename in os.listdir(folder):
            if filename.endswith(".mp4"):
                video_path = os.path.join(folder, filename)
                video, _, _ = read_video(video_path)
                if transform:
                    video = transform(video)
                videos.append(video)
                label = 0 if 'real' in folder else 1  # Assuming folder names indicate real/fake
                labels.append(label)
        return videos, labels

    train_videos_real, train_labels_real = load_videos_from_folder(data_dir_real, train_transform)
    train_videos_fake, train_labels_fake = load_videos_from_folder(data_dir_fake, train_transform)
    test_videos_real, test_labels_real = load_videos_from_folder(data_dir_real, test_transform)
    test_videos_fake, test_labels_fake = load_videos_from_folder(data_dir_fake, test_transform)

    train_videos = train_videos_real + train_videos_fake
    train_labels = train_labels_real + train_labels_fake
    test_videos = test_videos_real + test_videos_fake
    test_labels = test_labels_real + test_labels_fake

    train_dataset = data.TensorDataset(torch.stack(train_videos), torch.tensor(train_labels))
    test_dataset = data.TensorDataset(torch.stack(test_videos), torch.tensor(test_labels))

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, test_loader
