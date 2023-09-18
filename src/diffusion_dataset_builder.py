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
        for noise in range(num_noise_levels):
            t = noise / (num_noise_levels + 1)
            gamma = DiffusionData.get_noise_schedule_factor(t)
        
            new_image, noise = DiffusionData.forward_noise(image, gamma)

            self.images.append(new_image)
            self.noise_images.append(noise)
            image = new_image
    
    def get_noise_schedule_factor(t, start=0, end=1, tau=1, clip_min=1e-9):
        return 1-t
        #A gamma function based on cosine function.
        #v_start = np.cos(start * np.pi / 2) ** (2 * tau)
        #v_end = np.cos(end * np.pi / 2) ** (2 * tau)
        #output = np.cos((t * (end - start) + start) * np.pi / 2) ** (2 * tau)
        #output = (v_end - output) / (v_end - v_start)
        #return np.clip(output, clip_min, 1.)
    
    def forward_noise(image : Image, gamma : float):
        image_data = np.array(image)
        #image_data = image_data / 256.0
        epsilon = np.sqrt(1 - gamma) * np.uint8(np.random.random(size=image_data.shape) * 256.0)
        image_data = np.clip(np.sqrt(gamma) * image_data + epsilon, 0, 256.0)
        return Image.fromarray(np.uint8(image_data)), Image.fromarray(np.uint8(epsilon))
    
    def backward_noise(query : Image, predicted_noise : Image, denoise_time : float) -> Image:
        query_data = np.array(query)
        noise_data = np.array(predicted_noise)
        gamma = DiffusionData.get_noise_schedule_factor(denoise_time)
        y = np.clip(query_data - noise_data, 0, 256.0)
        x0 = y / np.sqrt(gamma)
        return Image.fromarray(np.uint8(x0))
    
    def generate_noise_image(size):
        image = Image.new('RGB', size)
        data = np.array(image)
        data = np.clip(np.random.randint(0, 256, size=data.shape), 0, 256)
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
            targets = torch.Tensor(size=(num_noise_levels, 3, 256, 256))
            queries = torch.Tensor(size=(num_noise_levels, 3, 256, 256))
            times = torch.Tensor(size=(num_noise_levels, 1))

            noise_levels = [i for i in range(num_noise_levels)]

            idx = 0
            for noise_level in noise_levels:
                ref_index = np.random.randint(0, len(self.target_images))
                ref = self.target_images[ref_index]

                query = ref.images[noise_level]
                target = ref.noise_images[noise_level]
                
                #query.show()
                #target.show()

                targets[idx] = transform(target)
                queries[idx] = transform(query)
                times[idx] = noise_level / (num_noise_levels-1)
                idx = idx + 1
            return queries.to('cuda:0'), times.to('cuda:0'), targets.to('cuda:0')

if __name__ == '__main__':
    diffusion_builder = DiffusionDatasetBuilder()
    diffusion_builder.load_single_images()
    #queries, targets = diffusion_builder.sample_noisy_images()

    data = diffusion_builder.target_images[0]
    data.generate_images_with_noise(8)

    for i in range(8):
        image = data.images[i]
        noise_image = data.noise_images[i]

        #image.show()
        #noise_image.show()

    for i in range(7):
        query = data.images[i]
        
        prediction = data.noise_images[i]
        backward_image = DiffusionData.backward_noise(query, prediction, i/(8-1))
        query.show()
        backward_image.show()


