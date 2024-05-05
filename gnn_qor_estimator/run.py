import train
from dataset_gpu import CustomDataset
import sys

with open("active_configs.txt", 'r') as file:
    # Read all lines into a list
    for line in file:
        line = line.strip()
        words = line.split(" ")

        dataset = CustomDataset(f"{words[1]}_{words[2]}")
        dataset = dataset.shuffle()
        
        print("Training on: " + dataset.saveDir)
        trainer = train.Train(dataset, line)
        trainer.train_main()



