# 输入一张图片，检测，按任意键退出
# 命令行 e.g. python image_detect.py -i demo/rs1.jpg -m fh02.pth

# encoding:utf-8
import cv2
import torch
from torch.autograd import Variable
import argparse
from load_data import *
from utils import PDR
import config


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to the input folder")
    ap.add_argument("-m", "--model", required=True, help="path to the model file")
    args = vars(ap.parse_args())

    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    batchSize = 8 if use_gpu else 8
    resume_file = args['model']
    img_path = args['input']

    img_width = config.img_width
    img_height = config.img_height

    provNum, alphaNum, adNum = config.provNum, config.alphaNum, config.adNum
    provinces = config.provinces
    alphabets = config.alphabets
    ads = config.ads

    imgSize = (img_width, img_height)

    def pre_detect(img_data,i,j):
        img_data = img_data[580*j:580*j+1160, i*360:i*360+720]
        cv2.imshow('cut', img_data)
        resizedImage = cv2.resize(img_data, imgSize)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        img_data = resizedImage
        img_data = img_data.reshape(1, 3, img_width, img_height)
        XI = torch.from_numpy(img_data)

        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        # Forward pass: Compute predicted y by passing x to the model
        box_pred, y_pred = model_conv(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]
        labelPreds = [max(t[0]) for t in outputY]
        print(labelPreds)
        lpn = alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + \
              ads[labelPred[6]]
        print(provinces[labelPred[0]] + lpn)
        cv2.waitKey(0)

    model_conv = PDR(provNum, alphaNum, adNum)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
    model_conv.eval()

    # img_data = img_Data(img_path, imgSize)
    img_data = cv2.imread(img_path)
    x, y = img_data.shape[0:2]
    img_test = cv2.resize(img_data, (int(y / 2), int(x / 2)))
    for i in range(0, 6):
        for j in range(0, 3):
            pre_detect(img_test, i, j)
    # 3456 4608 large.jpg
    # 分辨率 宽720 高 1160 通道RGB 3
    # img_data = img_data[0:1160, 0:720]
    # cv2.imshow('cut', img_data)
    resizedImage = cv2.resize(img_data, imgSize)
    resizedImage = np.transpose(resizedImage, (2, 0, 1))
    resizedImage = resizedImage.astype('float32')
    resizedImage /= 255.0
    img_data = resizedImage
    img_data = img_data.reshape(1, 3, img_width, img_height)
    XI = torch.from_numpy(img_data)

    if use_gpu:
        x = Variable(XI.cuda(0))
    else:
        x = Variable(XI)
    # Forward pass: Compute predicted y by passing x to the model
    box_pred, y_pred = model_conv(x)

    outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
    labelPred = [t[0].index(max(t[0])) for t in outputY]
    labelPreds = [max(t[0]) for t in outputY]
    print(labelPreds)

    [cx, cy, w, h] = box_pred.data.cpu().numpy()[0].tolist()

    # 从imgname读取
    img = cv2.imread(img_path)

    left_up = [(cx - w/2)*img.shape[1], (cy - h/2)*img.shape[0]]
    right_down = [(cx + w/2)*img.shape[1], (cy + h/2)*img.shape[0]]
    cv2.rectangle(img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])), (0, 0, 255), 2)
    # The first character is Chinese character, can not be printed normally, thus is omitted.
    lpn = alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + ads[labelPred[6]]
    cv2.putText(img, lpn, (int(left_up[0]), int(left_up[1])-20), cv2.FONT_ITALIC, 2, (0, 0, 255))
    print(provinces[labelPred[0]] + lpn)
    cv2.imshow('PD', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
