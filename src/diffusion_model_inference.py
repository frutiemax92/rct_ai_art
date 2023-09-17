from diffusion_model import RCTDiffusionModel
from diffusion_dataset_builder import DiffusionDatasetBuilder
import PIL.Image as Image
import PIL.ImageChops as ImageChops
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
        noise_image = self.dataset.generate_noise_image()
        #noise_image.show()

        x = torch.Tensor(size=(1, 3, 256, 256)).to('cuda:0')
        x[0] = transform(noise_image).to('cuda:0')

        time = torch.Tensor(size=(1,1)).to('cuda:0')
        to_pil = T.ToPILImage()
        
        transform2 = T.ToPILImage()
        with torch.no_grad():
            self.diffusion_model.eval()
            stop_iter = RCTDiffusionModel.denoise_levels
            for i in range(2):
                time[0] = (RCTDiffusionModel.denoise_levels - 1 - i)
                #time[0] = i
                predicted_noise = self.diffusion_model(x, time)
                predicted_noise = to_pil(predicted_noise[0])
                noise_image = ImageChops.subtract(noise_image, predicted_noise)
                #noise_image.show()
                x[0] = transform(noise_image).to('cuda:0')
        return transform2(x[0].to('cpu'))

if __name__ == '__main__':
    diffusion_model = RCTDiffusionModel().to('cuda:0')
    dataset = DiffusionDatasetBuilder()
    diffusion_model_inference = DiffusionModelInference(diffusion_model, dataset)
    diffusion_model_inference.load_model('model_weigths.pth')
    im = diffusion_model_inference.evaluate()
    im.save('out.png')