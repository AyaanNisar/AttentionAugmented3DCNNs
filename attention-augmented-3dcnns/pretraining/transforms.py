import torchio as tio

def get_simclr_augmentations():
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=2, p=0.5),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=3, p=0.3),
        tio.RandomNoise(std=(0, 0.05), p=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
        tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1, 2), p=0.2),
        tio.RandomMotion(degrees=5, translation=2, num_transforms=2, p=0.2),
        tio.RandomSwap(patch_size=8, num_iterations=2, p=0.1),
    ]) 