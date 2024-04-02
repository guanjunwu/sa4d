import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import glob
from lib.baselines import DINO, get_model
import torchvision.transforms as tfs
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
from sklearn.decomposition import PCA

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_dataloader(args):
    transform = get_default_transform(args.imsize)

    # Image dataset
    dataset = ImageFolderNoLabels(args.dir_images, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dataloader


def save(features, filenames, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_dict = dict()
    features = torch.Tensor(features.numpy()) # [N, C, H, W]
    for idx, f in enumerate(tqdm(filenames, desc='Saving')):
        features_dict[os.path.basename(f)] = features[idx, ...]
    torch.save(features_dict, output_path)


def get_default_transform(imsize):
    transform = tfs.Compose([
        # tfs.Resize(256),
        tfs.Resize(imsize),
        # tfs.CenterCrop(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=MEAN, std=STD)
    ])
    return transform


class ImageFolderNoLabels(VisionDataset):
    def __init__(self, root, transform, loader=default_loader, is_valid_file=None):
        root = os.path.abspath(root)
        super().__init__(root, transform)
        self.loader = loader
        samples = self.parse_dir(root, IMG_EXTENSIONS if is_valid_file is None else None)
        self.samples = samples


    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(os.path.join(self.root, path))
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample, self.samples[index]

    def __len__(self):
        return len(self.samples)

    def parse_dir(self, dir, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if not os.path.isdir(dir):
            raise IOError(f"{dir} is not a directory.")
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        # parse
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    images.append(path)
        return images

if __name__ == '__main__':
    parser = ArgumentParser(description="SAM feature extracting params")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--model", type=str, default="dino")
    parser.add_argument("--dir_images", default='cut_roasted_beef', type=str)
    parser.add_argument("--model_path", type=str, default="ckpts/dino_vitbase8_pretrain.pth")
    parser.add_argument("--imsize", default=(480, 676), nargs="+", type=int)
    args = parser.parse_args()
    
    print("Initializing DINO...")
    model = get_model(args.model, args.model_path, f"cuda:{args.gpu}")
    transform = get_default_transform(args.imsize)
     
    scene_type = "dynerf"
    videos = glob.glob(os.path.join(args.dir_images, "cam*.mp4"))
    videos = sorted(videos)
    eval_index =0
    all_features = []
    all_filenames = {}
    print('Extracting features...')
    # for index, video_path in enumerate(videos):
    #     if index == eval_index:
    #         continue
        
    #     camera_path = video_path.split('.')[0]
    #     image_path = os.path.join(camera_path, "images")   
    #     images_path = os.listdir(image_path)
    #     images_path.sort()
    #     all_filenames[camera_path] = []
    #     for path in tqdm(images_path):
    #         name = path.split('.')[0]
    #         img = default_loader(os.path.join(image_path, path))
    #         img = transform(img).unsqueeze(0)
    #         feature = model.extract_features(img, transform=False, upsample=False)
    #         all_features.append(feature.cpu())
    #         all_filenames[camera_path].extend(path)
    
    for index, video_path in enumerate(videos):
        if index == eval_index:
            continue
        if index != 4: continue
        camera_path = video_path.split('.')[0]
        image_path = os.path.join(camera_path, "images")
        args.dir_images = image_path
        dataloader = get_dataloader(args)
        all_filenames[camera_path] = []
        
        for batch, filenames in tqdm(dataloader):
            # print(batch.shape)
            # sys.exit(0)
            batch_feats = model.extract_features(batch, transform=False, upsample=False)
            all_filenames[camera_path].extend(filenames)
            all_features.append(batch_feats.detach().cpu())
            
    # print(all_filenames)
    all_features = torch.cat(all_features, 0)
    pca = PCA(n_components=args.pca)
    N, C, H, W = all_features.shape
    print("Features shape: ", all_features.shape)
    all_features = all_features.permute(0, 2, 3, 1).view(-1, C).numpy()
    print("Features shape: ", all_features.shape)
    # sys.exit(0)
    X = pca.fit_transform(all_features)
    print("Features shape (PCA): ", X.shape)
    X = torch.Tensor(X).view(N, H, W, args.pca).permute(0, 3, 1, 2)
    # output_path_pca = os.path.join(os.path.split(args.dir_images)[0], "features.pt")
    # print(f'Saving features to {output_path_pca}')
    
    for index, video_path in enumerate(videos):
        if index == eval_index:
            continue
        if index != 4: continue
        camera_path = video_path.split('.')[0]
        output_path_pca = os.path.join(camera_path, "features.pt")
        print(f'Saving features to {output_path_pca}')
        save(X, all_filenames[camera_path], output_path_pca)
        
                    
