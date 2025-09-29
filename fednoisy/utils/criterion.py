import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def linear_rampup(lambda_u, current, warm_up, rampup_length=16):
    """DivideMix"""
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class NegEntropy(object):
    """DivideMix"""

    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


class DivideMixSemiLoss(object):
    """DivideMix"""

    def __call__(
        self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(lambda_u, epoch, warm_up)


def loss_coteaching(outputs1, outputs2, noisy_label, forget_rate, noise_or_not):
    with torch.no_grad():
        loss1 = F.cross_entropy(outputs1, noisy_label, reduce=False)
        idx1_sorted = torch.argsort(loss1.data)
        loss1_sorted = loss1[idx1_sorted]

        loss2 = F.cross_entropy(outputs2, noisy_label, reduce=False)
        idx2_sorted = torch.argsort(loss2.data)
        loss2_sorted = loss1[idx2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss1_sorted))

        pure_ratio1 = noise_or_not[idx1_sorted[:num_remember]].sum() / num_remember
        pure_ratio2 = noise_or_not[idx2_sorted[:num_remember]].sum() / num_remember

        idx1_update = idx1_sorted[:num_remember]
        idx2_update = idx2_sorted[:num_remember]
    # exchange
    loss1_update = F.cross_entropy(
        outputs1[idx2_update], noisy_label[idx2_update], reduction="mean"
    )
    loss2_update = F.cross_entropy(
        outputs2[idx1_update], noisy_label[idx1_update], reduction="mean"
    )
    return loss1_update, loss2_update, pure_ratio1, pure_ratio2


"""Code for robust loss functions is from https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py"""


#### TODO: modified version with no device argument ####
def get_robust_loss(num_classes, args,alpha=0.1,reduction='mean'):
    if args.criterion == "ce":
        return nn.CrossEntropyLoss(reduction=reduction)
    elif args.criterion == "sce":
        return SCELoss(args.sce_alpha, args.sce_beta, num_classes)
    elif args.criterion == "tsce":
        return SCEteachLoss(args.sce_alpha, args.sce_beta)
    elif args.criterion == 'lsce':
        return SCELoss(alpha, args.sce_beta, num_classes)
    elif args.criterion == "rce":
        return ReverseCrossEntropy(num_classes, args.loss_scale,reduction=reduction)
    elif args.criterion == "nrce":
        return NormalizedReverseCrossEntropy(num_classes, args.loss_scale)
    elif args.criterion == "nce":
        return NormalizedCrossEntropy(num_classes, args.loss_scale)
    elif args.criterion == "gce":
        return GeneralizedCrossEntropy(num_classes, args.gce_q)
    elif args.criterion == "ngce":
        return NormalizedGeneralizedCrossEntropy(
            num_classes, args.loss_scale, args.gce_q
        )
    elif args.criterion == "mae":
        return MeanAbsoluteError(num_classes, args.loss_scale)
    elif args.criterion == "nmae":
        return NormalizedMeanAbsoluteError(num_classes, args.loss_scale)
    elif args.criterion == "focal":
        return FocalLoss(args.focal_gamma, args.focal_alpha)
    elif args.criterion == "nfocal":
        return NormalizedFocalLoss(
            args.loss_scale, args.focal_gamma, num_classes, args.focal_alpha
        )
    else:
        raise ValueError(
            f"args.criterion='{args.criterion}' is not supported. Only support 'ce', 'sce', 'rce', 'nrce', 'nce', 'gce', 'ngce', 'mae', 'nmae', 'focal', 'nfocal'."
        )


class SCELoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=10,reduction='mean'):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction


    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        if self.reduction == 'mean':
            rce = rce.mean()
        loss = self.alpha * ce + self.beta * rce
        return loss

class SCELoss_kd(nn.Module):
    def __init__(self, alpha, beta, num_classes=10,reduction='mean'):
        super(SCELoss_kd, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction


    def forward(self, pred, labels,soft_target,weight_kd):
        log_pred = torch.log_softmax(pred, dim=-1)
        # set the log_pred from inf small to 0
        log_pred = torch.where(torch.isinf(log_pred), torch.full_like(log_pred, -7), log_pred)
        kl_reduction = 'batchmean' if self.reduction == 'mean' else 'none'
        kl = F.kl_div(log_pred, soft_target, reduction=kl_reduction)

        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        if self.reduction == 'mean':
            rce = rce.mean()
        else:
            kl = torch.sum(kl,dim=1)
        loss = self.alpha * ce + self.beta * rce

        loss_kd = weight_kd * kl + (1-weight_kd)*loss
        return loss_kd

class LA_SCELoss_kd(nn.Module):
    def __init__(self,cls_num_list, alpha, beta, num_classes=10,reduction='mean',tau=1):
        super(LA_SCELoss_kd, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)


    def forward(self, pred, labels,soft_target,weight_kd):
        pred = pred + self.m_list
        log_pred = torch.log_softmax(pred, dim=-1)
        # set the log_pred from inf small to 0
        log_pred = torch.where(torch.isinf(log_pred), torch.full_like(log_pred, -7), log_pred)
        kl_reduction = 'batchmean' if self.reduction == 'mean' else 'none'
        kl = F.kl_div(log_pred, soft_target, reduction=kl_reduction)

        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        if self.reduction == 'mean':
            rce = rce.mean()
        else:
            kl = torch.sum(kl,dim=1)
        loss = self.alpha * ce + self.beta * rce

        loss_kd = weight_kd * kl + (1-weight_kd)*loss
        return loss_kd

class SCEesLoss(nn.Module):
    def __init__(self, num_classes=10):
        super(SCEesLoss, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred,labels,weight_kd):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = (1-weight_kd) * ce + weight_kd * rce.mean()
        return loss

class LA_SCEesLoss(nn.Module):
    def __init__(self, cls_num_list,num_classes=10):
        super(LA_SCEesLoss, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, pred,labels,weight_kd):
        # CE
        pred = pred+self.m_list
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = (1-weight_kd) * ce + weight_kd* rce.mean()
        return loss

class tSCEesLoss(nn.Module):
    def __init__(self, num_classes=10):
        super(tSCEesLoss, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, global_output,labels,weight_kd):
        # CCE
        ce = self.cross_entropy(pred, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        rce = self.cross_entropy(label_log,global_output)

        # Loss
        loss = weight_kd * ce + (1-weight_kd)* rce.mean()
        return loss

class tSCEesCE(nn.Module):
    def __init__(self, num_classes=10):
        super(tSCEesCE, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, global_output,labels,weight_kd):
        # CCE
        ce = self.cross_entropy(pred, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        global_output = F.softmax(global_output, dim=1)
        global_output = torch.clamp(global_output, min=1e-7, max=1.0)
        rce = -1 * torch.sum(global_output * label_log, dim=1)

        # Loss
        loss = (1-weight_kd) * ce + rce.mean()
        return loss

class tSCEes2Loss(nn.Module):
    def __init__(self, num_classes=10):
        super(tSCEes2Loss, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, global_output,labels,weight_kd):
        # CCE
        ce = self.cross_entropy(pred, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        rce = self.cross_entropy(label_log,global_output)

        # Loss
        loss = (1-weight_kd) * ce + weight_kd* rce.mean()
        return loss

class LA_SCELoss(nn.Module):
    def __init__(self, cls_num_list,alpha, beta, num_classes=10, reduction='mean'):
        super(LA_SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, pred, labels):
        # CCE
        pred = pred+self.m_list
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        if self.reduction == 'mean':
            rce =  rce.mean()
        loss = self.alpha * ce + self.beta * rce
        return loss


class SCEteachLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=10,reduction = 'mean'):
        super(SCEteachLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction = reduction)
        self.reduction = reduction

    def forward(self, pred, global_output,labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        global_output = F.softmax(global_output, dim=1)
        global_output = torch.clamp(global_output, min=1e-7, max=1.0)
        rce = -1 * torch.sum(global_output * label_log, dim=1)
        # rce = self.cross_entropy(label_log,global_output)
        if self.reduction == 'mean':
            rce = rce.mean()
        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

class LA_SCEteachLoss(nn.Module):
    def __init__(self, cls_num_list,alpha, beta, num_classes=10, tau=1,reduction='mean'):
        super(LA_SCEteachLoss, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction


    def forward(self, pred, global_output,labels):
        pred = pred + self.m_list
        # CCE
        ce = self.cross_entropy(pred, labels,reduction=self.reduction)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        global_output = global_output +self.m_list
        global_output = F.softmax(global_output, dim=1)
        global_output = torch.clamp(global_output, min=1e-7, max=1.0)
        rce = -1 * torch.sum(global_output * label_log, dim=1)
        # rce = self.cross_entropy(label_log,global_output)

        if self.reduction == 'mean':
            rce = rce.mean()

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

class LA_SCEteachLoss2(nn.Module):
    def __init__(self, cls_num_list,alpha, beta, num_classes=10, tau=1,reduction='mean'):
        super(LA_SCEteachLoss2, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction)
        self.reduction = reduction

    def forward(self, pred, global_output,labels):
        pred = pred + self.m_list
        # CCE
        ce = self.cross_entropy(pred, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        global_output = F.softmax(global_output, dim=1)
        global_output = torch.clamp(global_output, min=1e-7, max=1.0)
        rce = -1 * torch.sum(global_output * label_log, dim=1)
        # rce = self.cross_entropy(label_log,global_output)
        if self.reduction == 'mean':
            rce = rce.mean()
        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss


class LA_SCEteachesCE(nn.Module):
    def __init__(self,cls_num_list, num_classes=10):
        super(LA_SCEteachesCE, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, pred, global_output,labels,weight_kd):
        # CCE
        pred = pred+self.m_list
        ce = self.cross_entropy(pred, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        global_output = global_output +self.m_list
        global_output = F.softmax(global_output, dim=1)
        global_output = torch.clamp(global_output, min=1e-7, max=1.0)
        rce = -1 * torch.sum(global_output * label_log, dim=1)

        # Loss
        loss = (1-weight_kd) * ce + rce.mean()
        return loss


class LA_SCEteachesCE2(nn.Module):
    def __init__(self,cls_num_list, num_classes=10):
        super(LA_SCEteachesCE2, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, pred,global_output,labels,weight_kd):
        # CCE
        pred = pred+self.m_list
        ce = self.cross_entropy(pred, labels)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_log = torch.log(label_one_hot)
        # RCE
        global_output = F.softmax(global_output, dim=1)
        global_output = torch.clamp(global_output, min=1e-7, max=1.0)
        rce = -1 * torch.sum(global_output * label_log, dim=1)

        # Loss
        loss = (1-weight_kd) * ce + rce.mean()
        return loss



class ReverseCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0,reduction='mean'):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        if self.reduction == 'mean':
            return self.scale * rce.mean()
        else:
            return self.scale * rce



class NormalizedReverseCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        return self.scale * normalizor * rce.mean()


class NormalizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * nce.mean()


class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


class NormalizedGeneralizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        numerators = 1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


class MeanAbsoluteError(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()


class NormalizedMeanAbsoluteError(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        normalizor = 1 / (2 * (self.num_classes - 1))
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()


class NCEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


class NCEandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)


class GCEandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes=10, q=0.7):
        super(GCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.mae(pred, labels)


class GCEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.rce(pred, labels)


class GCEandNCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.nce(pred, labels)


class NGCEandNCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)


class NGCEandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)


class NGCEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(
            scale=alpha, q=q, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.rce(pred, labels)


class MAEandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(MAEandRCE, self).__init__()
        self.num_classes = num_classes
        self.mae = MeanAbsoluteError(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.mae(pred, labels) + self.rce(pred, labels)


class NLNL(nn.Module):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.0
        if not hasattr(train_loader.dataset, "targets"):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (
                    torch.from_numpy(np.array(train_loader.dataset.targets)) == i
                ).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (
            labels.unsqueeze(-1).repeat(1, self.ln_neg)
            + torch.LongTensor(len(labels), self.ln_neg).random_(1, self.num_classes)
        ) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes - 1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(
            labels
        ) * self.ln_neg

        s_neg = torch.log(torch.clamp(1.0 - F.softmax(pred, 1), min=1e-5, max=1.0))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size())
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(
            s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)
        ) * float((labels_neg >= 0).sum())
        loss = (loss + loss_neg) / (
            float((labels >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum())
        )
        return loss


class FocalLoss(nn.Module):
    """
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NormalizedFocalLoss(nn.Module):
    def __init__(
        self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True
    ):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NFLandNCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandNCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.nce(pred, labels)


class NFLandMAE(nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class NFLandRCE(nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes
        )
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class DMILoss(nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


def mixup_criterion(criterion, pred, y_a, y_b, lmbd):
    """We use cross-entropy loss as criterion here, so it's same as using criterion(pred, lmbd*y_a + (1-lmbd)*y_b)

    Args:
        criterion (_type_): Default as cross-entropy
        pred (_type_): _description_
        y_a (_type_): _description_
        y_b (_type_): _description_
        lmbd (_type_): _description_

    Returns:
        _type_: _description_
    """
    return lmbd * criterion(pred, y_a) + (1 - lmbd) * criterion(pred, y_b)
