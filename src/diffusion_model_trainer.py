from diffusion_model import RCTDiffusionModel
from diffusion_dataset_builder import DiffusionDatasetBuilder
import PIL.Image as Image
import torch
import numpy as np
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

class DiffusionModelTrainer:
    def __init__(self, diffusion_model : RCTDiffusionModel, dataset : DiffusionDatasetBuilder):
        self.diffusion_model = diffusion_model
        self.dataset = dataset
    
    def show_prediction(self, pred : torch.Tensor):
        transform = T.ToPILImage()
        pred_image = transform(pred.to('cpu'))
        pred_image.show()

    def train(self, num_epochs = 1000, learn_rate=1e-3, batch_size=32):
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=learn_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        self.diffusion_model.train()
        loss_values = []
        
        for t in range(num_epochs):
            queries, times, targets = self.dataset.sample_noisy_images(RCTDiffusionModel.denoise_levels, 'one_by_noise_level')
            for batch_count in range(0, queries.size(0), batch_size):
                end = np.minimum(batch_count + batch_size, queries.size(0))
                pred = self.diffusion_model(queries[batch_count:end], times[batch_count:end])

                #self.show_prediction(pred[61])
                #self.show_prediction(targets[61])

                loss = loss_fn(pred, targets[batch_count:end])
                loss.backward()
                loss_values.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            
            if t % 10 == 0:
                print(f't={t}, mean_loss = {np.mean(loss_values)}')
                loss_values.clear()

    def save_model(self, filename):
        torch.save(self.diffusion_model.state_dict(), filename)

if __name__ == '__main__':
    diffusion_model = RCTDiffusionModel().to('cuda:0')

    if os.path.exists('model_weigths.pth'):
        diffusion_model.load_state_dict(torch.load('model_weigths.pth'))

    dataset = DiffusionDatasetBuilder()
    dataset.load_single_images()

    diffusion_model_trainer = DiffusionModelTrainer(diffusion_model, dataset)
    diffusion_model_trainer.train(500, learn_rate=1e-3, batch_size=64)
    diffusion_model_trainer.save_model('model_weigths.pth')