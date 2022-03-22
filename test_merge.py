import os
import numpy as np
from ensemble_boxes import *
import torch
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])


    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

data_dir1 = './merge/fold1/'
data_dir2 = './merge/fold2/'
data_dir3 = './merge/fold3/'
data_dir4 = './merge/fold4/'
data_dir5 = './merge/fold5/'
ground_truth_path = './merge/label/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

name = os.listdir(data_dir1)
label1 = [os.path.join(data_dir1,file) for file in name]
label2 = [os.path.join(data_dir2,file) for file in name]
label3 = [os.path.join(data_dir3,file) for file in name]
label4 = [os.path.join(data_dir4,file) for file in name]
label5 = [os.path.join(data_dir5,file) for file in name]
ground_truth = [os.path.join(ground_truth_path,file) for file in name]
len(name)

stats = []
weights = [1,1,1,1,1]
iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1
for num in range(len(name)):
    temp1 = np.loadtxt(label1[num])
    temp2 = np.loadtxt(label2[num])
    temp3 = np.loadtxt(label3[num])
    temp4 = np.loadtxt(label4[num])
    temp5 = np.loadtxt(label5[num])
    target = np.loadtxt(ground_truth[num])
    boxes_list = [xywh2xyxy(temp1[:,1:5]).tolist(),xywh2xyxy(temp2[:,1:5]).tolist(),xywh2xyxy(temp3[:,1:5]).tolist(),xywh2xyxy(temp4[:,1:5]).tolist(),xywh2xyxy(temp5[:,1:5]).tolist()]
    scores_list = [temp1[:,5].tolist(),temp2[:,5].tolist(),temp3[:,5].tolist(),temp4[:,5].tolist(),temp5[:,5].tolist()]
    labels_list = [temp1[:,0].tolist(),temp2[:,0].tolist(),temp3[:,0].tolist(),temp4[:,0].tolist(),temp5[:,0].tolist()]
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    scores = scores.reshape(len(scores),1)
    labels = labels.reshape(len(labels),1)
    pred = np.concatenate((boxes,scores,labels),axis =1)
    pbox = torch.from_numpy(pred).to(device)
    nl = len(target)
    tbox = xywh2xyxy(target[:, 1:5])
    tcls = target[:,0]
    tbox = torch.from_numpy(tbox).to(device)
    tcls = torch.from_numpy(tcls).to(device)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    detected = []
    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
    for cls in torch.unique(tcls):
        ti = (cls == tcls).nonzero(as_tuple=False).view(-1)
        pi = (cls == pbox[:,5]).nonzero(as_tuple=False).view(-1)
        if pi.shape[0]:
            # Prediction to target ious
            ious, i = box_iou(pbox[pi, :4], tbox[ti]).max(1)  # best ious, indices

            # Append detections
            detected_set = set()
            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                d = ti[i[j]]  # detected target
                if d.item() not in detected_set:
                    detected_set.add(d.item())
                    detected.append(d)
                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                    if len(detected) == nl:  # all targets already located in image
                        break
    stats.append((correct.cpu(), pbox[:, 4].cpu(), pbox[:, 5].cpu(), tcls.cpu()))

stats = [np.concatenate(x, 0) for x in zip(*stats)]
p, r, ap, f1, ap_class = ap_per_class(*stats)
ap50, ap =  ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
print('mAP.5:'+ map50 +'/n')
print('mAP.5:95:'+ map )