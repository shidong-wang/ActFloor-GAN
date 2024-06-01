import os
import numpy as np

def getIDtxt(dataset, outputPath):
    if os.path.exists(outputPath):
        os.remove(outputPath)

    with open(outputPath, 'w') as file:
        for i in range(len(dataset)):
            id = dataset[i]
            if i == (len(dataset)-1):
                file.write(id)
            else:
                file.write(id + '\n')

    return 0
    
def main(datasetDir, outputDir):
    
    np.random.seed(None)
    dataset = np.array([pth_path.split('.')[0] for pth_path in os.listdir(datasetDir)])
    np.random.shuffle(dataset)

    num_train = 75000
    num_test = 2500

    dataset_train = dataset[:num_train]
    dataset_test = dataset[num_train:num_train+num_test]
    dataset_valid = dataset[num_train+num_test:len(dataset)]

    print(f' train: {len(dataset_train)} test: {len(dataset_test)} valid: {len(dataset_valid)}')
    
    getIDtxt(dataset_train, f'{outputDir}/train.txt')
    getIDtxt(dataset_test, f'{outputDir}/test.txt')
    getIDtxt(dataset_valid, f'{outputDir}/valid.txt')
    
    return 0

if __name__ == '__main__':
    datasetDir = 'dataset_4c'
    outputDir = 'dataset_split'
    main(datasetDir, outputDir)
