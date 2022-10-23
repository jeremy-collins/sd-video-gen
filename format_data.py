import os
import argparse

def format_data(folder):
    # adding the first 80% of the data to train and the rest to test
    folder_list = os.listdir(folder)
    # removing yaml files
    folder_list = [x for x in folder_list if not x.endswith('.yml')]
    folder_list.sort()
    train_list = folder_list[:int(len(folder_list)*0.8)]
    test_list = folder_list[int(len(folder_list)*0.8):]

    # creating the train and test folders
    if not os.path.exists(os.path.join(folder, 'train')):
        os.mkdir(os.path.join(folder, 'train'))
    if not os.path.exists(os.path.join(folder, 'test')):
        os.mkdir(os.path.join(folder, 'test'))

    # moving the folders to train and test
    for train_folder in train_list:
        os.rename(os.path.join(folder, train_folder), os.path.join(folder, 'train', train_folder))
    for test_folder in test_list:
        os.rename(os.path.join(folder, test_folder), os.path.join(folder, 'test', test_folder))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()
    # format_data('./data/complex_large_10_22_copy')
    format_data(args.folder)