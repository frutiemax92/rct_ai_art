from PIL import Image
from PIL import ImageChops
import os
import numpy as np
import torch
import torchvision.transforms as T

class DiffusionData:
    def __init__(self, img : Image):
        self.target = img
        self.images = []
        self.noise_images = []
    
    def generate_images_with_noise(self, num_noise_levels):
        image = self.target

        noise_image = DiffusionData.generate_noise_image(image.size)
        for noise in range(num_noise_levels):
            scale = self.get_scale(noise, num_noise_levels)
            new_image = ImageChops.blend(image, noise_image, scale)
            delta = ImageChops.subtract(new_image, image)
            self.images.append(new_image)
            self.noise_images.append(delta)
            image = new_image
    
    def get_scale(self, noise_level, num_noise_levels):
        if noise_level == 0:
            return 0.0
        return 2**noise_level/ 2**(num_noise_levels - 1)
    
    def generate_noise_image(size):
        image = Image.new('RGB', size)
        data = np.array(image)
        data = np.random.normal(128.0, 16.0, size=data.shape)
        return Image.fromarray(np.uint8(data))

    
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
                self.target_images.append(DiffusionData(image))
    
    def generate_noisy_images(self, num_noise_levels):
        for target in self.target_images:
            target.generate_images_with_noise(num_noise_levels)
        
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
            targets = torch.Tensor(size=(num_noise_levels - 1, 3, 256, 256))
            queries = torch.Tensor(size=(num_noise_levels - 1, 3, 256, 256))
            times = torch.Tensor(size=(num_noise_levels - 1, 1))

            noise_levels = [i for i in range(1, num_noise_levels)]

            idx = 0
            for noise_level in noise_levels:
                ref_index = np.random.randint(0, len(self.target_images))
                ref = self.target_images[ref_index]

                query = ref.images[noise_level]
                target = ref.noise_images[noise_level]
                
                targets[idx] = transform(target)
                queries[idx] = transform(query)
                times[idx] = noise_level
                idx = idx + 1
            return queries.to('cuda:0'), times.to('cuda:0'), targets.to('cuda:0')

if __name__ == '__main__':
    diffusion_builder = DiffusionDatasetBuilder()
    diffusion_builder.load_single_images()
    #queries, targets = diffusion_builder.sample_noisy_images()

    data = DiffusionData(diffusion_builder.target_images[0])
    images, noise_images = data.get_images_with_noise(8)

    for i in range(8):
        image = images[i]
        noise_image = noise_images[i]

        image.show()
        noise_image.show()


