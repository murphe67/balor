
from dataset import CustomDataset

with open("active_configs.txt", 'r') as file:
    # Read all lines into a list
    for line in file:
        line = line.strip()

        print("Generating dataset:", line)

        dataset = CustomDataset(line)
        dataset.generateDataset(True)