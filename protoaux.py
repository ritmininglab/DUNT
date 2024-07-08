"""Adapted from https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototransfer.utils_proto_spl import prototypical_loss, get_prototypes
#Protoaux_cls
class Protoaux(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?
    
    """
    def __init__(self, encoder, distance, device,n_support=1, n_query=1,spl=0.2):
        super(Protoaux, self).__init__()
        self.n_support = n_support
        self.n_query = n_query

        self.encoder = encoder
        self.device = device
        self.distance = distance
        self.spl=spl

    def loss(self, sample, ways,pri=False):
        # Extract support and query data
        # with shape [batch_size x num_samples x img_dims]
        # Labels are dummy labels in [0, ..., ways]
        if pri:
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
            # shots = int(x_support.size(1) / ways)
            # test_shots = int(x_query.size(1) / ways)
        else:
            data = sample['data'].to(self.device)  # [batch_size x ways x shots x image_dim]
            # data = batch['train'].to(self.device)
            data = data.unsqueeze(0)  # TODO remove when multi-batching is supported
            # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
            batch_size = data.size(0)
            ways = data.size(1)

            # Divide into support and query shots
            x_support = data[:, :, :self.n_support]
            x_support = x_support.reshape(
                (batch_size, ways * self.n_support, *x_support.shape[-3:]))  # e.g. [1,50*n_support,*(3,84,84)]
            x_query = data[:, :, self.n_support:]
            x_query = x_query.reshape(
                (batch_size, ways * self.n_query, *x_query.shape[-3:]))  # e.g. [1,50*n_query,*(3,84,84)]

            # Create dummy query labels
            y_query = torch.arange(ways).unsqueeze(0).unsqueeze(2)  # batch and shot dim
            y_query = y_query.repeat(batch_size, 1, self.n_query)
            y_query = y_query.view(batch_size, -1).to(self.device)

            y_support = torch.arange(ways).unsqueeze(0).unsqueeze(2)  # batch and shot dim
            y_support = y_support.repeat(batch_size, 1, self.n_support)
            y_support = y_support.view(batch_size, -1).to(self.device)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1)
        z = self.encoder.forward(x)
        z_support = z[:,:ways*self.n_support]
        z_query = z[:,ways*self.n_support:]

        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support  # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        if self.spl:
            loss, accuracy, loss_n, predictions = prototypical_loss(z_proto, z_query, y_query,
                                                                    distance=self.distance)

            threshold = sorted(loss_n[0])[-int(y_query.size(1) * self.spl)]  # omit 1/3 high loss query examples
            index = [i for i in range(loss_n.shape[1]) if loss_n[0][i] < threshold+0.001]
            y_query = y_query[:, index]
            z_query = z_query[:, index, :]

        loss, accuracy, loss_n, predictions = prototypical_loss(z_proto, z_query, y_query,
                                                                distance=self.distance)
        del loss_n, predictions

        return loss, accuracy


class Protoaux_fea(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, n_support, n_query, distance, device):
        super(Protoaux_fea, self).__init__()
        self.n_support = n_support
        self.n_query = n_query

        self.encoder = encoder
        self.device = device
        self.distance = distance

    def loss(self, sample, ways, pri=False):
        # Extract support and query data
        # with shape [batch_size x num_samples x img_dims]
        # Labels are dummy labels in [0, ..., ways]
        if pri:
            if "support" in sample.keys():
                x_support = sample["support"][0]
            else:
                x_support = sample["train"][0]
            x_support = x_support.to(self.device)

            if "query" in sample.keys():
                x_query = sample["query"][0]
            else:
                x_query = sample["test"][0]
            x_query = x_query.to(self.device)
        else:
            data = sample['data'].to(self.device)  # [batch_size x ways x shots x image_dim]
            # data = batch['train'].to(self.device)
            data = data.unsqueeze(0)  # TODO remove when multi-batching is supported
            # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
            batch_size = data.size(0)
            ways = data.size(1)

            # Divide into support and query shots
            x_support = data[:, :, :self.n_support]
            x_support = x_support.reshape(
                (batch_size, ways * self.n_support, *x_support.shape[-3:]))  # e.g. [1,50*n_support,*(3,84,84)]
            x_query = data[:, :, self.n_support:]
            x_query = x_query.reshape(
                (batch_size, ways * self.n_query, *x_query.shape[-3:]))  # e.g. [1,50*n_query,*(3,84,84)]


        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1)
        z = self.encoder.forward(x)

        return z

class Protoaux_dis(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, distance,device,dataset):
        super(Protoaux_dis, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance
        if dataset == 'cifar_fs':
            self.fc1 = nn.Linear(256, 50).to(self.device)
        elif dataset =='cub':
            self.fc1 = nn.Linear(1600, 50).to(self.device)
        else:
            self.fc1 = nn.Linear(1600, 50).to(self.device)
        self.fc2 = nn.Linear(50, 1).to(self.device)

    def loss(self, sample):
        # self.fc1.to(self.device)
        e1 = F.relu(self.fc1(sample))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
