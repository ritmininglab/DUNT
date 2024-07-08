"""Adapted from https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py"""

import torch
import torch.nn as nn

from utils_proto_spl import prototypical_loss, get_prototypes

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
        loss, accuracy,loss_n,predictions = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)
        del loss_n

        return loss, accuracy,predictions,[]



class Protonet_spl(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, distance='euclidean', device="cpu",spl=0.3):
        super(Protonet_spl, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance
        self.spl=spl

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
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query, y_query,
                                                   distance=self.distance)
        threshold = sorted(loss_n[0])[-int(y_query.size(1) * self.spl)]  # omit 1/3 high loss query examples
        index = [i for i in range(loss_n.shape[1]) if loss_n[0][i] < threshold]
        # index=[]
        # for x in range(loss_n.shape[1]):
        #     if loss_n[0][x] < threshold:
        #         index.append(x)
        y_query_spl = y_query[:, index]
        z_query_spl = z_query[:, index, :]
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query_spl, y_query_spl,
                                                   distance=self.distance)

        del loss_n
        return z,loss, accuracy,predictions,index




class Protonet_fsr(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, distance='euclidean', device="cpu",spl=0.9):
        super(Protonet_fsr, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance
        self.spl=spl

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
        loss, accuracy, loss_n_meta,predictions = prototypical_loss(z_support, z_query, y_query,
                                                   distance=self.distance)
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query, y_query,
                                                   distance=self.distance)
        meta_margin = loss_n-loss_n_meta
        threshold = sorted(meta_margin[0])[-int(y_query.size(1) * self.spl)]
        # omit 1/3 high loss query examples
        if threshold>0:
            index = [i for i in range(meta_margin.shape[1]) if meta_margin[0][i] < threshold]
        else:
            index = [i for i in range(meta_margin.shape[1]) ]
        # index=[]
        # for x in range(loss_n.shape[1]):
        #     if loss_n[0][x] < threshold:
        #         index.append(x)
        y_query_R = y_query[:, index]
        z_query_R = z_query[:, index, :]
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query_R, y_query_R,
                                                   distance=self.distance)

        del loss_n
        return loss, accuracy,predictions,index,z




class Protonet_head(nn.Module):
    """Prototypical network

    Parameters:
    - encoder (nn.Module): Embedding function.
    - distance (string): Use euclidean or cosine distance.
    - device (string, torch.device): Use GPU or CPU?

    """

    def __init__(self, encoder, distance='euclidean', device="cpu",spl=0.3,dataset='cifar_fs',num_class=30):
        super(Protonet_spl, self).__init__()
        self.encoder = encoder
        self.device = device
        self.distance = distance
        self.spl=spl
        if dataset == 'cifar_fs':
            self.head = nn.Linear(256, num_class).to(self.device)
        else:
            self.head=nn.Linear(1600, num_class).to(self.device)

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
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query, y_query,
                                                   distance=self.distance)
        threshold = sorted(loss_n[0])[-int(y_query.size(1) * self.spl)]  # omit 1/3 high loss query examples
        index = [i for i in range(loss_n.shape[1]) if loss_n[0][i] < threshold]
        # index=[]
        # for x in range(loss_n.shape[1]):
        #     if loss_n[0][x] < threshold:
        #         index.append(x)
        y_query_spl = y_query[:, index]
        z_query_spl = z_query[:, index, :]
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query_spl, y_query_spl,
                                                   distance=self.distance)

        del loss_n
        return loss, accuracy,predictions,index,z




