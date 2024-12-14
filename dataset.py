import torchvision.transforms as tvf

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader


transform = tvf.Compose(
    [tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
     tvf.ToTensor(),
     tvf.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])]
)


class MyDataset(Dataset):
    def __init__(self, image_path: Path):

        self.image_path_list = sorted(list(image_path.glob('*.png')))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        img = Image.open(img_path).convert('RGB')

        '''
        image: Tensor
        image path: tuple(str1, str2, ...)
        '''
        return (transform(img), index)
    
    
def main():
    dataset_dir = 'YOUR DATASET(IMAGES: .png) DIRECTTORY'
    path = Path(dataset_dir)
    dataset = CustomDataset(path)
    loader = DataLoader(dataset,
                        batch_size=5,
                        num_workers=0)
    
    for t_img, idx in loader:
        print(idx)


if __name__ == '__main__':
    main()