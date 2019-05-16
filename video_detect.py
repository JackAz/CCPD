# 从摄像头获取图片，不断循环检测，按 e 退出循环
# 命令行 e.g. python video_detect.py -i demo/rs1.jpg -m fh02.pth

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
    ap.add_argument("-m", "--model", required=True,help="path to the model file")
    args = vars(ap.parse_args())

    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    batchSize = 8 if use_gpu else 8
    resume_file = args['model']

    frame_width = 1280
    frame_height = 720
    img_width = config.img_width
    img_height = config.img_height

    provNum, alphaNum, adNum = config.provNum, config.alphaNum, config.adNum
    provinces = config.provinces
    alphabets = config.alphabets
    ads = config.ads

    imgSize = (img_width, img_height)

    # initialize Plate_Detect_Recognize model
    model_conv = PDR(provNum, alphaNum, adNum)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
    model_conv.eval()

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        img_data = video_Data(frame, imgSize)
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
        [cx, cy, w, h] = box_pred.data.cpu().numpy()[0].tolist()

        img = frame
        left_up = [(cx - w / 2) * img.shape[1], (cy - h / 2) * img.shape[0]]
        right_down = [(cx + w / 2) * img.shape[1], (cy + h / 2) * img.shape[0]]
        cv2.rectangle(img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])),
                      (0, 0, 255), 2)
        # The first character is Chinese character, can not be printed normally, thus is omitted.
        lpn = alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[
            labelPred[5]] + ads[labelPred[6]]
        cv2.putText(img, lpn, (int(left_up[0]), int(left_up[1]) - 20), cv2.FONT_ITALIC, 2, (0, 0, 255))
        print(provinces[labelPred[0]] + lpn)
        cv2.imshow('PDR', img)
        # 按 e 退出循环
        if cv2.waitKey(1) == ord('e'):
            break

    capture.release()
    cv2.destroyAllWindows()
