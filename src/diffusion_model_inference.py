from diffusion_model import RCTDiffusionModel
from diffusion_dataset_builder import DiffusionDatasetBuilder
import PIL.Image as Image
import torch
import numpy as np
import torchvision.transforms as T

class DiffusionModelInference:
    def __init__(self, diffusion_model : RCTDiffusionModel, dataset : DiffusionDatasetBuilder):
        self.diffusion_model = diffusion_model
        self.dataset = dataset
    
    def load_model(self, filename):
        self.diffusion_model.load_state_dict(torch.load(filename))
    
    def evaluate(self) -> Image.Image:
        # generate noise image
        transform = T.ToTensor()
        noise_image = transform(self.dataset.generate_noise_image()).to('cuda:0')

        x = torch.Tensor(size=(1, 3, 256, 256)).to('cuda:0')
        x[0] = noise_image

        time = torch.Tensor(size=(1,1)).to('cuda:0')
        
        transform2 = T.ToPILImage()
        with torch.no_grad():
            self.diffusion_model.eval()
            stop_iter = self.diffusion_model.denoise_levels
            for i in range(6):
                time[0] = (self.diffusion_model.denoise_levels - 1 - i)
                x = self.diffusion_model(x, time)
        
        return transform2(x[0].to('cpu'))

if __name__ == '__main__':
    diffusion_model = RCTDiffusionModel(8).to('cuda:0')
    dataset = DiffusionDatasetBuilder()
    diffusion_model_inference = DiffusionModelInference(diffusion_model, dataset)
    diffusion_model_inference.load_model('model_weigths.pth')
    im = diffusion_model_inference.evaluate()
    im.show()