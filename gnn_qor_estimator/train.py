import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from models import GNNDSE, Hier, ArchC
from inference import inference

import os


class Train:
    def __init__(self, dataset, configName):
        num_graphs = len(dataset)
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.dataset = dataset
        self.configName = configName

        generator1 = torch.Generator().manual_seed(42)

        self.trainData, self.valData, self.testData = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15],generator=generator1)

        self.trainData = self.trainData[0:-1]
        self.valData = self.valData[0:-1]
        self.testData = self.testData[0:-1]

        print(f'{num_graphs} graphs in total:')
        print(f'train: {len(self.trainData)}, val:  {len(self.valData)}, test: {len(self.testData)}')
        batchSize = 16
        self.train_loader = DataLoader(self.trainData, batch_size=batchSize, shuffle=True)
    

        self.num_features = self.train_loader.dataset[0].num_features
        self.edge_dim = self.train_loader.dataset[0].edge_attr.shape[1]

        configSplit = configName.split(" ")
        modelType = configSplit[0]
        actualConfig = configSplit[1]
        name = configSplit[2]
        self.model_dir = f"model_weights/{actualConfig}/{modelType}/{name}/"
        self.results_dir = f"results/{actualConfig}/{modelType}/{name}/"
        self.training_error_dir = f"training_error/{actualConfig}/{modelType}/{name}/"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.training_error_dir, exist_ok=True)

        features = self.num_features
        edges_dim = self.edge_dim
        if modelType == "gnndse":
            self.model = GNNDSE(6, features,edges_dim).to(self.device)
        if modelType == "gnndse2":
            self.model = GNNDSE(2,features, edges_dim).to(self.device)
        elif modelType == "hier":
            self.model = Hier(features, edges_dim).to(self.device)
        elif modelType == "archC":
            self.model = ArchC(features, edges_dim).to(self.device)
    


    def train_main(self):
        epochs = range(1000)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        train_losses = []
        for epoch in epochs:
            print(f'Epoch {epoch + 1} train')
            loss, loss_dict_train = train(epoch, self.model, self.train_loader, optimizer, self.device)

            print((f'Train loss breakdown {loss_dict_train}'))

            train_losses.append(loss)

            if (epoch + 1) % 10 == 0:
                self.save(epoch, train_losses)


    def save(self, epoch, train_losses):
        torch.save(self.model.state_dict(),f"{self.model_dir}/{epoch}.pth")
        epochs = range(epoch+1)

        import matplotlib.pyplot as plt
        plt.plot(epochs, train_losses, 'g', label='Training loss')

        plt.title('Training, Validation, and Testing loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.training_error_dir}/{epoch}.png")
        plt.clf()

        inference(self, self.valData, "val", epoch)
        inference(self, self.testData, "test", epoch)

def train(epoch, model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    loss_dict = {}
    target_list = ["LUTs", "FFs", "BRAMs", "DSPs", "clock", "latency"]
    for t in target_list:
        loss_dict[t] = 0.0

    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        out, loss, loss_dict_ = model.to(device)(data)
        loss.backward()

        total_loss += loss.item() * data.num_graphs


        for t in target_list:
            loss_dict[t] += loss_dict_[t].item()
        
        optimizer.step()
    
    return total_loss / len(train_loader.dataset), {key: v / len(train_loader) for key, v in loss_dict.items()}
   
