from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob


# Calculate mean and std of training data
class MyDataset(Dataset):
    def __init__(self):
        self.imgspath = []
        for imname in glob.glob('../data/train/*.png'):
            # Run in all image in folder
            self.imgspath.append(imname)

        print('Total data: {}'.format(len(self.imgspath)))

    def __getitem__(self, index):
        imgpath = self.imgspath[index]
        image = Image.open(imgpath).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        image = transform(image)
        return image

    def __len__(self):
        return len(self.imgspath)


dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.

for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print("Mean: {}".format(mean))
print("Std: {}".format(std))
