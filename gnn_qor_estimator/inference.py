from torch_geometric.loader import DataLoader
from ucla_models import Net
import torch
from dataset_gpu import CustomDataset
from tqdm import tqdm
import csv


def unnormalize(input, max):
    return int(((input+1)/2) * max)

def inference(configName, epoch, dataset, outName):

    loader = DataLoader(dataset, shuffle=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = loader.dataset[0].num_features
    edge_dim = loader.dataset[0].edge_attr.shape[1]

    device = torch.device('cpu')
    model = Net(6, 64, num_features, edge_dim=edge_dim).to(device)
    

    model.load_state_dict(torch.load(f"model_weights/{configName}/{epoch}.pth"))
    with open(f"results/{configName}/{outName}_{epoch}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["real luts", "inferred luts", "real ffs", "inferred ffs", "real BRAMs", "inferred BRAMs", "real DSPs", "inferred DSPs"])
        with torch.no_grad():
            for data in tqdm(loader):  # Assuming you have a dataloader for inference
                data = data.to(device)

                # Forward pass
                out, loss, loss_dict_ = model.to(device)(data)

                csv_row = []

                for i, target in enumerate(["LUTs", "FFs", "BRAMs", "DSPs"]):
                    ground_truth = data.y[:, i][0]
                    csv_row.append(unnormalize(ground_truth, data.max_y[target]))
                    inferred = out[target][0][0]
                    csv_row.append(unnormalize(inferred, data.max_y[target]))
                writer.writerow(csv_row)




