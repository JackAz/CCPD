# 测试集上测试模型的准确率
# 命令行 e.g. python Eval.py -m fh02.pth -i D:\Dataset\CCPD2019\ccpd_test -s failure_save
# encoding:utf-8
import torch
from torch.autograd import Variable
import argparse
from os import path, mkdir
from load_data import *
from time import time
import config
from utils import PDR

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to the input folder")
    ap.add_argument("-m", "--model", required=True, help="path to the model file")
    ap.add_argument("-s", "--store", required=True, help="path to the store folder")
    args = vars(ap.parse_args())

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    numClasses = config.numClasses
    numPoints = config.numPoints
    imgSize = (config.img_width, config.img_height)
    batchSize = 8 if use_gpu else 4
    print("batch size:", batchSize)
    resume_file = str(args["model"])

    provNum, alphaNum, adNum = config.provNum, config.alphaNum, config.adNum
    provinces = config.provinces
    alphabets = config.alphabets
    ads = config.ads


    def isEqual(labelGT, labelP):
        # print (labelGT)
        # print (labelP)
        compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
        # print(sum(compare))
        return sum(compare)


    model_conv = PDR(provNum, alphaNum, adNum)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
    model_conv.eval()

    # efficiency evaluation
    # dst = imgDataLoader([args["input"]], imgSize)
    # trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)
    #
    # start = time()
    # for i, (XI) in enumerate(trainloader):
    #     x = Variable(XI.cuda(0))
    #     y_pred = model_conv(x)
    #     outputY = y_pred.data.cpu().numpy()
    #     #   assert len(outputY) == batchSize
    # print("detect efficiency %s seconds" %(time() - start))

    count = 0
    correct = 0
    error = 0
    sixCorrect = 0
    sFolder = str(args["store"])
    sFolder = sFolder if sFolder[-1] == '/' else sFolder + '/'
    if not path.isdir(sFolder):
        mkdir(sFolder)

    dst = labelTestDataLoader(args["input"].split(','), imgSize)
    trainloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=1)
    with open('fh0Eval', 'wb') as outF:
        pass

    start = time()
    for i, (XI, labels, ims) in enumerate(trainloader):
        count += 1
        YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model_conv(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        #   compare YI, outputY
        # try:
        if isEqual(labelPred, YI[0]) == 7:
            correct += 1
            sixCorrect += 1
        else:
            sixCorrect += 1 if isEqual(labelPred, YI[0]) == 3 else 0

        if count % 50 == 0:
            print('total %s correct %s error %s precision %s three %s avg_time %s' % (
                count, correct, error, float(correct) / count, float(sixCorrect) / count, (time() - start) / count))
    with open('fh0Eval', 'a') as outF:
        outF.write('total %s correct %s error %s precision %s avg_time %s' % (
            count, correct, error, float(correct) / count, (time() - start) / count))
