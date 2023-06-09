import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
from scipy import stats
from scipy.optimize import curve_fit
from loss import L2RankLoss
from model.evaluator import GMS_3DQA
from data.datasets import QMM_Dataset

def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def main(config,i_fold):
    config_dict = vars(config)
    for key in config_dict:
        print(key,':',config_dict[key])
    
    print('-'*30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.model_type == 'swin':
        model = GMS_3DQA(checkpoint=config.load_path)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)


    #database configuration
    datainfo = os.path.join('datainfo',config.database,config.database + '_datainfo_' + str(i_fold) + '.csv')
    images_dir = config.images_dir
    print('using datainfo: ' + datainfo)

    
    if config.loss == 'l2rank':
        criterion = L2RankLoss().to(device)
        print('Using l2rank loss')
    else:
        criterion = nn.MSELoss().to(device)

    # dataloader configuration
    trainset = QMM_Dataset(csv_file = datainfo, data_prefix = images_dir, phase='train',img_length_read=config.img_length_read)
    testset = QMM_Dataset(csv_file = datainfo, data_prefix = images_dir, phase='test',img_length_read=config.img_length_read)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

    
    



    best_test_criterion = -1  # accuracy
    best_test = []
    print('Starting training:')

    for epoch in range(config.epochs):
        
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        start = time.time()
        for i, data in enumerate(train_loader):
            image = data['image'].to(device)
            labels = data['gt_label'].float().detach().to(device)
            
            outputs = model(image)
            optimizer.zero_grad()
            

            loss = criterion(outputs,labels)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()
            
            optimizer.step()
        


        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss),flush=True)

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))
        end = time.time()
        print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end-start))

        # do validation after each epoch
        start = time.time()
        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset)])
            y_pred = np.zeros([len(testset)])
            for i, data in enumerate(test_loader):
                    
                image = data['image'].to(device)
                label[i] = data['gt_label'].item()
                outputs = model(image)

                y_pred[i] = outputs.item()
               

        
            y_pred = fit_function(label,y_pred)
            test_PLCC = stats.pearsonr(y_pred, label)[0]
            test_SRCC = stats.spearmanr(y_pred, label)[0]
            test_KRCC = stats.kendalltau(y_pred, label)[0]
            test_RMSE = np.sqrt(((y_pred-label) ** 2).mean())
            end = time.time()
            
            print('Epoch %d testing time cost: %.4f seconds' % (epoch + 1, end-start))
            if test_SRCC > best_test_criterion:               
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1),flush=True)
                print('Updataed SRCC: {:.4f}, PLCC: {:.4f}, KRCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC, test_PLCC, test_KRCC, test_RMSE),flush=True)
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_PLCC, test_KRCC, test_RMSE]
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                #torch.save(model.state_dict(), os.path.join(config.ckpt_path, config.database + '_fold_' + str(i_fold) + '_best.pth'))
    return best_test
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_type', type=str, default = 'swin')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--images_dir', type=str, default='path-to-the-projections')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--ckpt_path', type=str, default='trained_ckpt')
    parser.add_argument('--load_path', type=str, default='checkpoint/swin_tiny_patch4_window7_224_22k.pth')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--img_length_read', type=int, default=6)
    

    
    config = parser.parse_args()
    set_rand_seed(seed=2023)
    results =  np.empty(shape=[0,4])
    for i_fold in range(config.k_fold):
        best_test = main(config,i_fold)
        print('--------------------------------------------The {}-th Fold-----------------------------------------'.format(i_fold+1))
        print('Training completed.')
        print('The best training result SRCC: {:.4f}, PLCC: {:.4f}, KRCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test[0], best_test[1], best_test[2], best_test[3]))
        results = np.concatenate((results, np.array([best_test])), axis=0)
        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')

    print ('==============================done==============================================')
    print ('The mean best result:', np.mean(results, axis=0))
    print ('The median best result:', np.median(results, axis=0))