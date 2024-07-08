"""Adapted from https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototransfer.utils_proto import prototypical_loss, get_prototypes

class Protonet(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?
    
    """
    def __init__(self, encoder, distance='euclidean', device="cpu"):
        super(Protonet, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance

    def loss(self, sample, ways):
        # Extract support and query data
        # with shape [batch_size x num_samples x img_dims]
        # Labels are dummy labels in [0, ..., ways]
        if "support" in sample.keys():
            x_support = sample["support"][0]
            y_support = sample["support"][1]
        else:
            x_support = sample["train"][0]
            y_support = sample["train"][1]
        x_support = x_support.to(self.device)
        y_support = y_support.to(self.device)

        if "query" in sample.keys():
            x_query = sample["query"][0]
            y_query = sample["query"][1]
        else:
            x_query = sample["test"][0]
            y_query = sample["test"][1]
        x_query = x_query.to(self.device)
        y_query = y_query.to(self.device)

        # Extract shots
        shots = int(x_support.size(1) / ways)
        test_shots = int(x_query.size(1) / ways)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1)
        z = self.encoder.forward(x)
        z_support = z[:,:ways*shots]
        z_query = z[:,ways*shots:]

        # Calucalte prototypes
        z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)

        return loss, accuracy


class Protonet_fea(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, distance='euclidean', device="cpu"):
        super(Protonet_fea, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance

    def loss(self, sample, ways):
        # Extract support and query data
        # with shape [batch_size x num_samples x img_dims]
        # Labels are dummy labels in [0, ..., ways]
        if "support" in sample.keys():
            x_support = sample["support"][0]
            y_support = sample["support"][1]
        else:
            x_support = sample["train"][0]
            y_support = sample["train"][1]
        x_support = x_support.to(self.device)
        y_support = y_support.to(self.device)

        if "query" in sample.keys():
            x_query = sample["query"][0]
            y_query = sample["query"][1]
        else:
            x_query = sample["test"][0]
            y_query = sample["test"][1]
        x_query = x_query.to(self.device)
        y_query = y_query.to(self.device)

        # Extract shots
        shots = int(x_support.size(1) / ways)
        test_shots = int(x_query.size(1) / ways)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1)
        z = self.encoder.forward(x)
        z_support = z[:, :ways * shots]
        z_query = z[:, ways * shots:]

        # Calucalte prototypes
        z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)

        return z


class Protonet_cls(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, distance='euclidean', device="cpu"):
        super(Protonet_cls, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance

    def loss(self, sample, ways):
        # Extract support and query data
        # with shape [batch_size x num_samples x img_dims]
        # Labels are dummy labels in [0, ..., ways]
        if "support" in sample.keys():
            x_support = sample["support"][0]
            y_support = sample["support"][1]
        else:
            x_support = sample["train"][0]
            y_support = sample["train"][1]
        x_support = x_support.to(self.device)
        y_support = y_support.to(self.device)

        if "query" in sample.keys():
            x_query = sample["query"][0]
            y_query = sample["query"][1]
        else:
            x_query = sample["test"][0]
            y_query = sample["test"][1]
        x_query = x_query.to(self.device)
        y_query = y_query.to(self.device)

        # Extract shots
        shots = int(x_support.size(1) / ways)
        test_shots = int(x_query.size(1) / ways)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1)
        z = self.encoder.forward(x)
        z_support = z[:, :ways * shots]
        z_query = z[:, ways * shots:]

        # Calucalte prototypes
        z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)

        return loss


class Protonet_dis(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, distance,device):
        super(Protonet_dis, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance
        self.fc1 = nn.Linear(256, 50)
        self.fc2 = nn.Linear(50, 1)

    def loss(self, sample, ways):
        # # Extract support and query data
        # # with shape [batch_size x num_samples x img_dims]
        # # Labels are dummy labels in [0, ..., ways]
        # if "support" in sample.keys():
        #     x_support = sample["support"][0]
        #     y_support = sample["support"][1]
        # else:
        #     x_support = sample["train"][0]
        #     y_support = sample["train"][1]
        # x_support = x_support.to(self.device)
        # y_support = y_support.to(self.device)
        #
        # if "query" in sample.keys():
        #     x_query = sample["query"][0]
        #     y_query = sample["query"][1]
        # else:
        #     x_query = sample["test"][0]
        #     y_query = sample["test"][1]
        # x_query = x_query.to(self.device)
        # y_query = y_query.to(self.device)
        #
        # # Extract shots
        # shots = int(x_support.size(1) / ways)
        # test_shots = int(x_query.size(1) / ways)
        #
        # # Extract features (first dim is batch dim)
        # x = torch.cat([x_support, x_query], 1)
        # z = self.encoder.forward(x)
        # ways=5
        # shots=1
        z=sample
        # z_support = z[:, :ways * shots]
        # z_query = z[:, ways * shots:]
        #
        # # Calucalte prototypes
        # z_proto = get_prototypes(z_support, y_support, ways)
        #
        # # Calculate loss and accuracies
        # loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
        #                                    distance=self.distance)

        e1 = F.relu(self.fc1(z.cpu()))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
