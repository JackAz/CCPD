# Compared to fh0.py
# fh02.py remove the redundant ims in model input
# 训练车牌检测网络，读取Bbox.pth参数
# 命令行 e.g.
# python train_rpnet.py -i D:\Dataset\CCPD2019\ccpd_base_bbox -b 5 -se 0 -f rpnet -t D:\Dataset\CCPD2019\ccpd_test

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import argparse
from time import time
from load_data import *
from torch.optim import lr_scheduler
import config
from utils import PDR
from utils import get_n_params

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="path to the input file")
    ap.add_argument("-n", "--epochs", default=10000,
                    help="epochs for train")
    ap.add_argument("-b", "--batch", default=5,
                    help="batch size for train")
    ap.add_argument("-se", "--start_epoch", required=True,
                    help="start epoch for train")
    ap.add_argument("-t", "--test", required=True,
                    help="dirs for test")
    ap.add_argument("-r", "--resume", default='111',
                    help="file for re-train")
    ap.add_argument("-f", "--folder", required=True,
                    help="folder to store model")
    ap.add_argument("-w", "--writeFile", default='fh02.out',
                    help="file for output")
    args = vars(ap.parse_args())

    wR2Path = 'wR2.pth'
    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    numClasses = config.numClasses
    numPoints = config.numPoints
    imgSize = (config.img_width, config.img_height)
    # lpSize = (128, 64)
    provNum, alphaNum, adNum = config.provNum, config.alphaNum, config.adNum
    batchSize = int(args["batch"]) if use_gpu else 2
    trainDirs = args["images"].split(',')
    testDirs = args["test"].split(',')
    modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'
    storeName = modelFolder + 'fh03.pth'
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)

    epochs = int(args["epochs"])
    #   initialize the output file
    if not os.path.isfile(args['writeFile']):
        with open(args['writeFile'], 'wb') as outF:
            pass

    epoch_start = int(args["start_epoch"])
    resume_file = str(args["resume"])
    if not resume_file == '111':
        # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
        if not os.path.isfile(resume_file):
            print("fail to load existed model! Existing ...")
            exit(0)
        print("Load existed model! %s" % resume_file)
        model_conv = PDR(provNum, alphaNum, adNum)
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv.load_state_dict(torch.load(resume_file))
        model_conv = model_conv.cuda()
    else:
        model_conv = PDR(provNum, alphaNum, adNum, wR2Path)
        if use_gpu:
            model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
            model_conv = model_conv.cuda()

    print(model_conv)
    print(get_n_params(model_conv))

    criterion = nn.CrossEntropyLoss()
    # optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

    dst = labelFpsDataLoader(trainDirs, imgSize)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=8)
    lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)


    def isEqual(labelGT, labelP):
        compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
        # print(sum(compare))
        return sum(compare)


    def eval(model, test_dirs):
        count, error, correct = 0, 0, 0
        dst = labelTestDataLoader(test_dirs, imgSize)
        testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=8)
        start = time()
        for i, (XI, labels, ims) in enumerate(testloader):
            count += 1
            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            if use_gpu:
                x = Variable(XI.cuda(0))
            else:
                x = Variable(XI)
            # Forward pass: Compute predicted y by passing x to the model

            fps_pred, y_pred = model(x)

            outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
            labelPred = [t[0].index(max(t[0])) for t in outputY]

            #   compare YI, outputY
            try:
                if isEqual(labelPred, YI[0]) == 7:
                    correct += 1
                else:
                    pass
            except:
                error += 1
        return count, correct, error, float(correct) / count, (time() - start) / count


    def train_model(model, criterion, optimizer, num_epochs=25):
        # since = time.time()
        for epoch in range(epoch_start, num_epochs):
            lossAver = []
            model.train(True)
            lrScheduler.step()
            start = time()

            for i, (XI, Y, labels, ims) in enumerate(trainloader):
                if not len(XI) == batchSize:
                    continue

                YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
                Y = np.array([el.numpy() for el in Y]).T
                if use_gpu:
                    x = Variable(XI.cuda(0))
                    y = Variable(torch.FloatTensor(Y).cuda(0), requires_grad=False)
                else:
                    x = Variable(XI)
                    y = Variable(torch.FloatTensor(Y), requires_grad=False)
                # Forward pass: Compute predicted y by passing x to the model

                try:
                    fps_pred, y_pred = model(x)
                except:
                    continue

                # Compute and print loss
                loss = 0.0
                loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
                loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])
                for j in range(7):
                    l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                    loss += criterion(y_pred[j], l)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                try:
                    lossAver.append(loss.data[0])
                except:
                    pass

                if i % 50 == 1:
                    with open(args['writeFile'], 'a') as outF:
                        print('train %s images, use %s seconds, loss %s\n' % (i * batchSize, time() - start,
                                                                              sum(lossAver) / len(lossAver) if len(
                                                                                  lossAver) > 0 else 'NoLoss'))
                        outF.write('train %s images, use %s seconds, loss %s\n' % (i * batchSize, time() - start,
                                                                                   sum(lossAver) / len(lossAver) if len(
                                                                                       lossAver) > 0 else 'NoLoss'))
                    torch.save(model.state_dict(), storeName)
            print('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
            model.eval()
            count, correct, error, precision, avgTime = eval(model, testDirs)
            with open(args['writeFile'], 'a') as outF:
                outF.write('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
                outF.write('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
            torch.save(model.state_dict(), storeName + str(epoch))
        return model


    model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)
