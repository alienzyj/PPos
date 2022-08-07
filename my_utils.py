import copy
import random

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.special import comb
from skimage.filters import roberts
from skimage.measure import label
from sklearn import metrics
from torch.nn import functional as F


class TanimotoLoss(nn.Module):
    def __init__(self, gamma):
        super(TanimotoLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        batch_loss = 0

        for i in range(inputs.size(0)):
            loss = 1 - (self.cal_coefficient(inputs[i], targets[i]) + self.cal_coefficient(1 - inputs[i],
                                                                                           1 - targets[i])) / 2
            batch_loss = batch_loss + loss * ((1 - dice_coef(inputs[i], targets[i])) ** self.gamma)

        return batch_loss / inputs.size(0)

    def cal_coefficient(self, input, target, smooth=1):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = input_flat * target_flat
        return (intersection.sum() + smooth) / (
                (input_flat ** 2).sum() + (target_flat ** 2).sum() - intersection.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        pt = (1 - inputs) * targets + inputs * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy(inputs, targets, reduction='none') * focal_weight

        return torch.mean(loss)


class InverseWeightingFocalLoss(nn.Module):
    def __init__(self, gamma=2, correction=1):
        super(InverseWeightingFocalLoss, self).__init__()
        self.gamma = gamma
        self.correction = correction

    def forward(self, inputs, targets):
        focal_weight = torch.zeros_like(inputs)
        for i in range(targets.size(0)):
            if torch.sum(targets[i]).item() > 0:
                num_element = targets[i].numel() + self.correction
                num_nodule = torch.sum(targets[i]).item() + self.correction
                background_weight = num_element / 2 / (num_element - num_nodule)
                foreground_weight = num_element / 2 / num_nodule
                background_weight, foreground_weight = background_weight / (
                        background_weight + foreground_weight), foreground_weight / (
                                                               background_weight + foreground_weight)
            else:
                foreground_weight = 1
                background_weight = 1
            focal_weight[i] = foreground_weight * targets[i] + background_weight * (1 - targets[i])
        pt = (1 - inputs) * targets + inputs * (1 - targets)
        loss = F.binary_cross_entropy(inputs, targets, reduction='none') * focal_weight * pt.pow(self.gamma)

        return torch.mean(loss)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, threshold=0.1, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, contrastive_method='simclr'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.threshold = threshold
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            if self.contrastive_method == 'gcl':
                mask = torch.eq(labels, labels.T).float().to(device)
            elif self.contrastive_method == 'pcl':
                mask = (torch.abs(
                    labels.T.repeat(batch_size, 1) - labels.repeat(1, batch_size)) < self.threshold).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def dice_coef(inputs, targets, smooth=1):
    input_flat = inputs.view(-1)
    target_flat = targets.view(-1)
    intersection = input_flat * target_flat
    return (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


def dice_loss(inputs, targets):
    return 1 - dice_coef(inputs, targets)


def normalize(image):
    max_bound = {'lung': 400, 'mediastinum': 240}
    min_bound = {'lung': -1024, 'mediastinum': -160}
    lung_image = (image - min_bound['lung']) / (max_bound['lung'] - min_bound['lung'])
    lung_image[lung_image > 1] = 1
    lung_image[lung_image < 0] = 0
    mediastinal_image = (image - min_bound['mediastinum']) / (max_bound['mediastinum'] - min_bound['mediastinum'])
    mediastinal_image[mediastinal_image > 1] = 1
    mediastinal_image[mediastinal_image < 0] = 0

    return np.stack([lung_image, mediastinal_image]).astype(np.float32)


def my_iou(threshold=0.5, batch=False):
    if batch:
        def iou(inputs, targets):
            result = torch.tensor(0)
            inputs = inputs > threshold
            for i in range(len(inputs)):
                intersection = torch.sum(targets[i] * inputs[i])
                union = torch.sum(targets[i]) + torch.sum(inputs[i]) - intersection
                if union.item() == 0:
                    result = result + torch.tensor(1)
                else:
                    result = result + intersection.float() / union

            return result.item() / len(inputs)
    else:
        def iou(inputs, targets):
            inputs = inputs > threshold
            intersection = torch.sum(targets * inputs)
            union = torch.sum(targets) + torch.sum(inputs) - intersection
            if union.item() == 0:
                result = torch.tensor(1)
            else:
                result = intersection.float() / union

            return result.item()

    return iou


def my_roc(inputs, targets):
    auc = metrics.roc_auc_score(torch.flatten(targets), torch.flatten(inputs))

    return auc


@torch.no_grad()
def unet_predict(sample, resize_shape, unet):
    unet.eval()
    shape = sample.shape
    sample = normalize(sample)
    sample = ndi.zoom(sample, (1, 1, resize_shape[0] / shape[1], resize_shape[1] / shape[2]), order=2, mode="nearest")
    slices = voxel2pixel(sample)
    preds = torch.zeros(shape[0], 1, 256, 512, dtype=torch.float32)
    for i in range(slices.shape[0] // 64):
        _, _, _, _, _, preds[i * 64 + 3:(i + 1) * 64 + 3] = unet(torch.from_numpy(slices[i * 64:(i + 1) * 64]))
    if slices.shape[0] % 64 != 0:
        _, _, _, _, _, preds[-(slices.shape[0] % 64) - 3:-3] = unet(torch.from_numpy(slices[-(slices.shape[0] % 64):]))
    preds = preds[:, 0, :, :].numpy()
    preds = ndi.zoom(preds, (1, shape[1] / resize_shape[0], shape[2] / resize_shape[1]), order=2, mode="nearest")

    return preds


@torch.no_grad()
def unet_classifier_predict(sample, unet_classifier, crop_shape):
    shape = sample.shape
    unet_classifier.eval()
    if shape[0] < crop_shape[0]:
        sample = np.pad(sample, ((0, crop_shape[0] - shape[0]), (0, 0), (0, 0)), 'edge')
    if shape[1] < crop_shape[1]:
        sample = np.pad(sample, ((0, 0), (0, crop_shape[1] - shape[1]), (0, 0)), 'edge')
    if shape[2] < crop_shape[2]:
        sample = np.pad(sample, ((0, 0), (0, 0), (0, crop_shape[2] - shape[2])), 'edge')
    sample = np.pad(sample, ((0, (64 - sample.shape[0] % 64) % 64), (0, (64 - sample.shape[1] % 64) % 64),
                             (0, (64 - sample.shape[2] % 64) % 64)),
                    'edge')
    sample = normalize(sample)

    _, _, _, _, pred, _ = unet_classifier(torch.from_numpy(sample[np.newaxis, ...]))

    return pred[0, 0, :shape[0], :shape[1], :shape[2]].detach().numpy()


@torch.no_grad()
def classifier_predict(sample, classifier, resize_shape):
    classifier.eval()
    sample = normalize(sample)
    sample = torch.from_numpy(sample[np.newaxis, ...])
    sample = F.interpolate(sample, resize_shape, mode='trilinear', align_corners=True)
    label = classifier(sample)

    return label.item()


def post_process(sample, mask: np.ndarray, classifier, threshold, scale, resize_shape):
    pred = mask
    mask = mask > threshold

    for i in range(mask.shape[0]):
        if np.max(mask[i]) == 1:
            edges = roberts(mask[i])
            mask[i] = ndi.binary_fill_holes(edges)
            labels = label(mask[i], background=0)
            vals = np.unique(labels)
            vals = vals[vals != 0]
            if vals.size > 1:
                probability = np.zeros(vals.shape)
                for j, val in enumerate(vals):
                    probability[j] = np.mean(pred[i][labels == val])
                max_index = probability.argmax()
                mask[i][labels != vals[max_index]] = 0
            if np.sum(mask[i]) <= 20:
                mask[i] = 0

    labels = label(mask, background=0)
    vals = np.unique(labels)
    vals = vals[vals != 0]
    if vals.size > 1:
        probability = np.zeros(vals.size)
        for k, val in enumerate(vals):
            indices = np.where(labels == val)
            start = [0, 0, 0]
            end = [0, 0, 0]
            for i in range(3):
                start[i] = np.min(indices[i])
                end[i] = np.max(indices[i]) + 1
                scale_length = round((end[i] - start[i]) * scale)
                midpoint = (end[i] + start[i]) // 2
                if midpoint - scale_length // 2 < 0:
                    start[i] = 0
                    if scale_length > mask.shape[i]:
                        end[i] = mask.shape[i]
                    else:
                        end[i] = scale_length
                elif midpoint - scale_length // 2 + scale_length > mask.shape[i]:
                    end[i] = mask.shape[i]
                    if end[i] - scale_length < 0:
                        start[i] = 0
                    else:
                        start[i] = end[i] - scale_length
                else:
                    start[i] = midpoint - scale_length // 2
                    end[i] = start[i] + scale_length
            probability[k] = np.mean(pred[labels == val]) + classifier_predict(
                sample[start[0]:end[0], start[1]:end[1], start[2]:end[2]], classifier, resize_shape)
        max_index = probability.argmax()
        mask[labels != vals[max_index]] = 0

    return mask.astype(np.uint8)


def hausdorff_distance(inputs, targets):
    hausdorff_computer = sitk.HausdorffDistanceImageFilter()
    hausdorff_computer.Execute(sitk.GetImageFromArray(inputs), sitk.GetImageFromArray(targets))

    return hausdorff_computer.GetHausdorffDistance()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def random_flip(axis, prob):
    def func(x):
        if isinstance(axis, int):
            if random.random() < prob:
                x = np.flip(x, axis=axis)
        else:
            for a in axis:
                if random.random() < prob:
                    x = np.flip(x, axis=a)
        return x

    return func


def gaussian_blur(sigma, mode):
    def func(x):
        sig = random.uniform(*sigma)
        for i in range(x.shape[0]):
            x[i] = ndi.gaussian_filter(x[i], sig, mode=mode)
        return x

    return func


def random_crop_resize(shape, mode, crop_scale=None):
    def func(x):
        if crop_scale is not None:
            height = round(x.shape[-2] * random.uniform(*crop_scale))
            width = round(x.shape[-1] * random.uniform(*crop_scale))
            top = random.randint(0, x.shape[-2] - height)
            left = random.randint(0, x.shape[-1] - width)
            x = x[..., top:top + height, left:left + width]
        return ndi.zoom(x, (1, shape[-2] / x.shape[-2], shape[-1] / x.shape[-1]), mode=mode)

    return func


def random_shift(translation, mode):
    return lambda x: ndi.shift(x, (
    0, random.randint(-translation[0], translation[0]), random.randint(-translation[1], translation[1]),
    random.randint(-translation[2], translation[2])), mode=mode)


def random_rotate(degree, axis, mode, reshape):
    return lambda x: ndi.rotate(x, random.randint(-degree, degree), axis, mode=mode, reshape=reshape)


def get_module_device(module):
    return next(module.parameters()).device


@torch.no_grad()
def fine_segmentation(sample, mask, transformer, threshold, crop_shape):
    shape = mask.shape
    if crop_shape[0] > shape[0]:
        sample = np.pad(sample, ((0, crop_shape[0] - shape[0]), (0, 0), (0, 0)), 'edge')
        mask = np.pad(mask, ((0, crop_shape[0] - mask.shape[0]), (0, 0), (0, 0)), 'edge')
    if crop_shape[1] > shape[1]:
        sample = np.pad(sample, ((0, 0), (0, crop_shape[1] - shape[1]), (0, 0)), 'edge')
        mask = np.pad(mask, ((0, 0), (0, crop_shape[1] - mask.shape[1]), (0, 0)), 'edge')
    if crop_shape[2] > shape[2]:
        sample = np.pad(sample, ((0, 0), (0, 0), (0, crop_shape[2] - shape[2])), 'edge')
        mask = np.pad(mask, ((0, 0), (0, 0), (0, crop_shape[2] - mask.shape[2])), 'edge')

    if np.sum(mask) != 0:
        indices = mask.nonzero()
        start = [0, 0, 0]
        end = [0, 0, 0]
        for i in range(3):
            start[i] = np.min(indices[i])
            end[i] = np.max(indices[i]) + 1
            midpoint = (end[i] + start[i]) // 2
            scale_length = crop_shape[i]
            if midpoint - scale_length // 2 < 0:
                start[i] = 0
                if scale_length > mask.shape[i]:
                    end[i] = mask.shape[i]
                else:
                    end[i] = scale_length
            elif midpoint - scale_length // 2 + scale_length > mask.shape[i]:
                end[i] = mask.shape[i]
                if end[i] - scale_length < 0:
                    start[i] = 0
                else:
                    start[i] = end[i] - scale_length
            else:
                start[i] = midpoint - scale_length // 2
                end[i] = start[i] + scale_length
        pred = unet_classifier_predict(sample[start[0]:end[0], start[1]:end[1], start[2]:end[2]], transformer,
                                            crop_shape)
        pred = np.pad(pred, ((start[0], mask.shape[0] - end[0] + 1), (start[1], mask.shape[1] - end[1] + 1),
                             (start[2], mask.shape[2] - end[2] + 1)), 'constant', constant_values=0)
        pred = pred[:shape[0], :shape[1], :shape[2]]
        mask = pred > threshold
        for i in range(mask.shape[0]):
            if np.max(mask[i]) == 1:
                edges = roberts(mask[i])
                mask[i] = ndi.binary_fill_holes(edges)
                labels = label(mask[i], background=0)
                vals = np.unique(labels)
                vals = vals[vals != 0]
                if vals.size > 1:
                    probability = np.zeros(vals.shape)
                    for j, val in enumerate(vals):
                        probability[j] = np.mean(pred[i][labels == val])
                    max_index = probability.argmax()
                    mask[i][labels != vals[max_index]] = 0
                if np.sum(mask[i]) <= 20:
                    mask[i] = 0
    return mask.astype(np.uint8)


def voxel2pixel(voxels):
    shape = voxels.shape[1:]
    pixels = np.empty((shape[0] - 6, 14, *shape[1:]), np.float32)
    for i in range(3, shape[0] - 3):
        pixels[i - 3] = np.concatenate((voxels[0, i - 3:i + 4], voxels[1, i - 3:i + 4]))

    return pixels


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob, dimension):
    if random.random() >= prob:
        return x

    if dimension == "2D":
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        _, img_rows, img_cols = x.shape
        num_block = 10000
        for i in range(len(x)):
            for _ in range(num_block):
                block_noise_size_x = random.randint(1, img_rows // 10)
                block_noise_size_y = random.randint(1, img_cols // 10)
                noise_x = random.randint(0, img_rows - block_noise_size_x)
                noise_y = random.randint(0, img_cols - block_noise_size_y)
                window = orig_image[i, noise_x:noise_x + block_noise_size_x,
                         noise_y:noise_y + block_noise_size_y
                         ]
                window = window.flatten()
                np.random.shuffle(window)
                window = window.reshape((block_noise_size_x,
                                         block_noise_size_y))
                image_temp[i, noise_x:noise_x + block_noise_size_x,
                noise_y:noise_y + block_noise_size_y] = window
        local_shuffling_x = image_temp
    elif dimension == "3D":
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        _, img_rows, img_cols, img_deps = x.shape
        num_block = 10000
        for i in range(len(x)):
            for _ in range(num_block):
                block_noise_size_x = random.randint(1, img_rows // 10)
                block_noise_size_y = random.randint(1, img_cols // 10)
                block_noise_size_z = random.randint(1, img_deps // 10)
                noise_x = random.randint(0, img_rows - block_noise_size_x)
                noise_y = random.randint(0, img_cols - block_noise_size_y)
                noise_z = random.randint(0, img_deps - block_noise_size_z)
                window = orig_image[i, noise_x:noise_x + block_noise_size_x,
                         noise_y:noise_y + block_noise_size_y,
                         noise_z:noise_z + block_noise_size_z,
                         ]
                window = window.flatten()
                np.random.shuffle(window)
                window = window.reshape((block_noise_size_x,
                                         block_noise_size_y,
                                         block_noise_size_z))
                image_temp[i, noise_x:noise_x + block_noise_size_x,
                noise_y:noise_y + block_noise_size_y,
                noise_z:noise_z + block_noise_size_z] = window
        local_shuffling_x = image_temp
    else:
        raise ValueError("Please specify the correct input dimension")

    return local_shuffling_x


def image_in_painting(x, dimension):
    if dimension == "2D":
        _, img_rows, img_cols = x.shape
        cnt = 5
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
            block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y) * 1.0
            cnt -= 1
    elif dimension == "3D":
        _, img_rows, img_cols, img_deps = x.shape
        cnt = 5
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
            block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
            block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
    else:
        raise ValueError("Please specify the correct input dimension")
    return x


def image_out_painting(x, dimension):
    if dimension == "2D":
        _, img_rows, img_cols = x.shape
        image_temp = copy.deepcopy(x)
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2]) * 1.0
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y]
        cnt = 4
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                    noise_y:noise_y + block_noise_size_y]
            cnt -= 1
    elif dimension == "3D":
        _, img_rows, img_cols, img_deps = x.shape
        image_temp = copy.deepcopy(x)
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt = 4
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                    noise_y:noise_y + block_noise_size_y,
                                                    noise_z:noise_z + block_noise_size_z]
            cnt -= 1
    else:
        raise ValueError("Please specify the correct input dimension")

    return x


def painting(x, paint_rate, outpaint_rate, dimension):
    if random.random() < paint_rate:
        if random.random() < outpaint_rate:
            x = image_out_painting(x, dimension)
        else:
            x = image_in_painting(x, dimension)
    return x


def pixel_acc(inputs, targets):
    return np.sum(inputs == targets) / targets.size


def pixel_precision_recall_f1(inputs, targets):
    inputs = inputs.astype(np.bool)
    tp = np.sum(inputs * targets)
    fp = np.sum(inputs * (1 - targets))
    fn = np.sum((~inputs) * targets)
    if tp + fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1
    else:
        recall = tp / (tp + fn)
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def elastic_transform(alpha, sigma):
    """Elastic deformation of images as described in [Simard2003].

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    def func(image, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        d, h, w = image.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        random_alpha = random.uniform(*alpha)
        random_sigma = random.uniform(*sigma)
        dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), random_sigma, mode="constant", cval=0) * random_alpha
        dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), random_sigma, mode="constant", cval=0) * random_alpha
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distored_image = [map_coordinates(image[i, :, :], indices, order=1, mode='reflect') for i in range(d)]
        distored_image = np.concatenate(distored_image, axis=0)

        return distored_image.reshape(image.shape)

    return func

