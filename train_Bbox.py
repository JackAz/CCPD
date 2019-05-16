# 训练bbox预定位网络
# 命令行 e.g. python train_Bbox.py -i D:\Dataset\CCPD2019\ccpd_base_bbox

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
from utils import Bbox
from utils import get_n_params

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to the input file")
    ap.add_argument("-n", "--epochs", default=2, help="epochs for train")
    ap.add_argument("-b", "--batch", default=4, help="batch size for train")
    ap.add_argument("-r", "--resume", default='111', help="file for re-train")
    ap.add_argument("-w", "--writeFile", default='Bbox.out', help="file for output")
    args = vars(ap.parse_args())

    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    numPoints = config.numPoints
    imgSize = (config.img_width, config.img_height)

    batchSize = int(args["batch"]) if use_gpu else 8
    modelFolder = 'Bbox/'
    storeName = modelFolder + 'Bbox.pth'
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)

    epochs = int(args["epochs"])
    #   initialize the output file
    with open(args['writeFile'], 'wb') as outF:
        pass

    epoch_start = 0
    resume_file = str(args["resume"])
    if not resume_file == '111':
        # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
        if not os.path.isfile(resume_file):
            print("fail to load existed model! Existing ...")
            exit(0)
        print("Load existed model! %s" % resume_file)
        model_conv = Bbox(numPoints)
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv.load_state_dict(torch.load(resume_file))
        model_conv = model_conv.cuda()
    else:
        model_conv = Bbox(numPoints)
        if use_gpu:
            model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
            model_conv = model_conv.cuda()

    print("model:", model_conv)
    print("model params:", get_n_params(model_conv))

    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model_conv.parameters())
    # optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

    optimizer = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    dst = ChaLocDataLoader(args["images"].split(','), imgSize)
    train_loader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)


    def train_model(model, optimizer, num_epochs=5):
        # since = time.time()
        for epoch in range(epoch_start, num_epochs):
            lossAver = []
            model.train(True)
            lrScheduler.step()
            start = time()

            for i, (XI, YI) in enumerate(train_loader):
                # print('%s/%s %s' % (i, times, time()-start))
                YI = np.array([el.numpy() for el in YI]).T
                if use_gpu:
                    x = Variable(XI.cuda(0))
                    y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
                else:
                    x = Variable(XI)
                    y = Variable(torch.FloatTensor(YI), requires_grad=False)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(x)

                # Compute and print loss
                loss = 0.0
                if len(y_pred) == batchSize:
                    loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2])
                    loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
                    lossAver.append(loss.data[0])
                    # 输出当前误差
                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.save(model.state_dict(), storeName)
                if i % 50 == 1:
                    print(('train %s images, use %s seconds, loss %s\n' % (
                        i * batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:]))))
                    with open(args['writeFile'], 'a') as outF:
                        outF.write('train %s images, use %s seconds, loss %s\n' % (
                            i * batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
            print('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
            with open(args['writeFile'], 'a') as outF:
                outF.write('Epoch: %s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
            torch.save(model.state_dict(), storeName + str(epoch))
        return model


    model_conv = train_model(model_conv, optimizer, num_epochs=epochs)
