from PIL import Image
from PIL import ImageChops
import os
import numpy as np
import torch
import torchvision.transforms as T

class DiffusionDatasetBuilder:
    def __init__(self):
        self.target_images = []

    def load_single_images(self):
        # load all the images in the images folder
        files = os.listdir('images')

        for image_folder in files:
            file_path = os.path.join('images', image_folder)
            if os.path.isfile(file_path) == True:
                continue

            images_paths = [os.path.join(file_path, f'out{i}.png') for i in range(4)]
            for image_path in images_paths:
                image = Image.open(image_path)

                # only use the top left quadrant
                image = image.crop((0, 0, 256, 256))
                self.target_images.append(image)
        
    def sample_noisy_images(self, num_noise_levels : int = 8, algorithm='one_by_target') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # create noise images for every target image
        transform = T.ToTensor()
        inverse = T.ToPILImage()
        if algorithm == 'one_by_target':
            targets = torch.Tensor(size=(len(self.target_images), 3, 256, 256))
            queries = torch.Tensor(size=(len(self.target_images), 3, 256, 256))
            times = torch.Tensor(size=(len(self.target_images), 1))
            
            for ref_index in range(len(self.target_images)):
                ref = self.target_images[ref_index]
                noise_level = np.random.randint(0, num_noise_levels)
                noise_image = self.generate_noise_image(noise_level)

                target = noise_image
                query = ImageChops.add(noise_image, ref)
                
                targets[ref_index] = transform(target)
                queries[ref_index] = transform(query)
                times[ref_index] = noise_level
            return queries.to('cuda:0'), times.to('cuda:0'), targets.to('cuda:0')
        elif algorithm == 'one_by_noise_level':
            targets = torch.Tensor(size=(num_noise_levels, 3, 256, 256))
            queries = torch.Tensor(size=(num_noise_levels, 3, 256, 256))
            times = torch.Tensor(size=(num_noise_levels, 1))

            noise_levels = [i for i in range(num_noise_levels)]

            idx = 0
            for noise_level in noise_levels:
                ref_index = np.random.randint(0, len(self.target_images))
                ref = self.target_images[ref_index]
                noise_factor = self.get_noise_level(noise_level, num_noise_levels)
                noise_image = self.generate_noise_image(noise_factor)
                target = noise_image
                query = ImageChops.add(noise_image, ref)
                
                targets[idx] = transform(target)
                queries[idx] = transform(query)
                times[idx] = noise_level
                idx = idx + 1
            return queries.to('cuda:0'), times.to('cuda:0'), targets.to('cuda:0')
    
    def get_noise_level(self, noise_level, num_noise_levels) -> float:
      return 1.5 ** (noise_level) / (1.5 ** (num_noise_levels - 1))
    def generate_noise_image(self, noise_factor=1.0) -> Image.Image:
        if noise_factor == 0:
            return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        noise = np.random.randint(0, 256 * noise_factor, size=(256, 256, 3), dtype=np.uint8)
        noise_image = Image.fromarray(np.uint8(noise))
        return noise_image

if __name__ == '__main__':
    diffusion_builder = DiffusionDatasetBuilder()
    diffusion_builder.load_single_images()
    queries, targets = diffusion_builder.sample_noisy_images()

