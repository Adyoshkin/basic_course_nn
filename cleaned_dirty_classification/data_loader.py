from torchvision import transforms, datasets
from torch.utils import data
from config import *

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
train_transforms = transforms.Compose([
    transforms.ColorJitter(0.5, 0.3, 0.3, 0.5),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(size=IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)])

val_transforms = transforms.Compose([
    transforms.CenterCrop(size=IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)])

train_dataset = datasets.ImageFolder(TRAIN_DIR, train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, val_transforms)
test_dataset = ImageFolderWithPaths(TEST_DIR, val_transforms)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)

test_dataloader = data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)