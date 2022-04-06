import preprocess as pre
import idrid_dataset as idrid
import pandas as pd
import model
import torch
import run
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)

def test_model_2(model, size):
    path = 'idrid_datast/500/'

    test_lebel = pd.read_csv('idrid_datast/test.csv')
    data_test = idrid.get_test_dataset_2_classes(path, test_lebel, size)
    run.test(data_test, model)

def test_model_4(model, size):
    path = '/home/thiago/PycharmProjects/datasets/IDRI/500/'

    test_lebel = pd.read_csv('/home/thiago/PycharmProjects/datasets/IDRI/test.csv')
    data_test = idrid.get_test_dataset_4_classes(path, test_lebel, size)
    run.test(data_test, model)

# binary
def do_binary():
    path = 'idrid_datast/500/'
    data_lebel = pd.read_csv('idrid_datast/train.csv')
    class_weights = idrid.get_weight(data_lebel, n_classes=2)
    x = 1024
    train, valid = idrid.get_data_loader_2_classes(path, data_lebel, x)
    classificador = model.get_densenet121_2_classes()



    #path_loader = torch.load('models/binary_model.pt')
    #classificador.load_state_dict(path_loader)


    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(classificador.parameters(), lr=LR)

    train_losses, valid_losses = run.optimize(train, valid, classificador, criterion, optimizer, EPOCHS)

    epochs = range(EPOCHS)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, valid_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    test_model_2(classificador, x)
    torch.save(classificador.state_dict(), 'models/binary_model.pt')


def do_mc():
    path = 'idrid_datast/500/'

    x = 512

    data_lebel = pd.read_csv('idrid_datast/train.csv')
    data_augmentation = pd.read_csv('augmentation.csv')
    class_weights = idrid.get_weight(data_lebel, n_classes=4, graph=True, data_aug=data_augmentation)

    train, valid = idrid.get_data_loader_4_classes(path, data_lebel, x, aug = data_augmentation,  aug_path='augmentation')
    classificador = model.get_densenet121_mc()

    path_loader = torch.load('models/model_mc_aug_plus.pt')
    classificador.load_state_dict(path_loader)


    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(classificador.parameters(), lr=LR)

    print(train)

    train_losses, valid_losses = run.optimize(train, valid, classificador, criterion, optimizer, EPOCHS)

    epochs = range(EPOCHS)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, valid_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    test_model_4(classificador, x)
    torch.save(classificador.state_dict(), 'models/model_mc_aug_plus.pt')



def test_full_models(model_binary, model_mc, limit, size_1, size_2):
    path = 'idrid_datast/500/'
    test_lebel = pd.read_csv('idrid_datast/test.csv')

    data_test_binary = idrid.get_test_dataset(path, test_lebel, 1024)
    data_test_mc = idrid.get_test_dataset(path, test_lebel, 512)

    run.test_full(data_test_binary, data_test_mc, model_binary, model_mc, limit)


if __name__ == '__main__':

    # # test x = 300
    # pre.preprocessing("idrid_datast/test/")
    # pre.preprocessing("idrid_datast/train/")
    # idrid.do_augmentation('idrid_datast/train.csv', 'idrid_datast/500/train/')

    LR = 0.0001
    BATCH = 15
    EPOCHS = 100
    do_binary()

    LR = 0.0001
    BATCH = 15
    EPOCHS = 70
    do_mc()

    classificador_binary = model.get_densenet121_2_classes()
    path_loader = torch.load('models/binary_model.pt')
    classificador_binary.load_state_dict(path_loader)

    classificador_mc = model.get_densenet121_mc()
    path_loader = torch.load('models/model_mc.pt')
    classificador_mc.load_state_dict(path_loader)

    # test_model_2(classificador_binary, 1024)
    # test_model_4(classificador_mc, 512)
    test_full_models(classificador_binary, classificador_mc, 1, 1024, 512)
