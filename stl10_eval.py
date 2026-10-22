import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
from torchvision import transforms

warnings.filterwarnings("ignore")


def inference_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066),(0.2603, 0.2566, 0.2713))
    ])
    

def logistic_regression(embeddings, labels, embeddings_val, labels_val):
    X_train, X_test = embeddings, embeddings_val
    y_train, y_test = labels, labels_val

    clf = LogisticRegression(max_iter=100)
    clf = CalibratedClassifierCV(clf)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy STL10: ", acc)
    return acc


class STL10Eval:

    def __init__(self, image_size=224, device="cpu", root=None):
        self.device = device

        # root: directory containing stl10_binary/ (default: the original
        # GPU-machine path so every existing caller is unchanged; the
        # optional argument and the STL10_ROOT env var exist so other
        # machines can point at their local copy)
        if root is None:
            root = os.environ.get(
                "STL10_ROOT",
                "/home/jonoki/research/master/byol_trial/ExtreMA_stl10/data")

        transform = inference_transforms(img_size=(image_size, image_size))
        train_ds = torchvision.datasets.STL10(root,
                                              split='train',
                                              transform=transform,
                                              download=False)
        val_ds = torchvision.datasets.STL10(root,
                                            split='test',
                                            transform=transform,
                                            download=False)

        self.train_loader = DataLoader(train_ds, batch_size=64, num_workers=2)
        self.val_loader = DataLoader(val_ds, batch_size=64, num_workers=2)

    @torch.inference_mode
    def evaluate(self, model):
        # model = ijepa_model.target_encoder
        # model = ijepa_model.context_encoder
        embeddings, labels = self._get_image_embs_labels(
            model, self.train_loader)
        embeddings_val, labels_val = self._get_image_embs_labels(
            model, self.val_loader)
        acc = logistic_regression(embeddings, labels, embeddings_val, labels_val)
        return acc

    @torch.inference_mode
    def _get_image_embs_labels(self, model, dataloader):
        embs, labels = [], []
        for _, (images, targets) in enumerate(dataloader):
            with torch.no_grad():
                images = images.to(self.device)
                out = model(images)
                features = out.cpu().detach()
                features = features.mean(dim=1)
                embs.extend(features.tolist())
                labels.extend(targets.cpu().detach().tolist())
        return np.array(embs), np.array(labels)
