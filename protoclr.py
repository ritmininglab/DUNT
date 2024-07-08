"""Contrastive loss forumulation"""
import torch
import torch.nn as nn

from prototransfer.utils_proto import (prototypical_loss,
                                       get_prototypes)
import torch.nn.functional as F
class ProtoCLR(nn.Module):
    """Calculate the ProtoCLR loss on a batch of images.

    Parameters:
        - encoder: The encoder network.
        - device: The device to use.
        - n_support: The number of support shots.
        - n_query: The number of query shots.
    """
    def __init__(self, encoder, device, n_support=1, n_query=1,
                 distance='euclidean'):
        super(ProtoCLR, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        
        self.encoder = encoder
        self.device = device
        self.distance = distance

    def forward(self, batch):
        # Treat the first dim as way, the second as shots
        # and use the first shot as support (like 1-shot setting)
        # the remaining shots as query
        data = batch['data'].to(self.device) # [batch_size x ways x shots x image_dim]
        #data = batch['train'].to(self.device)
        data = data.unsqueeze(0) #TODO remove when multi-batching is supported
        # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
        batch_size = data.size(0)
        ways = data.size(1)
        
        # Divide into support and query shots 
        x_support = data[:,:,:self.n_support]
        x_support = x_support.reshape((batch_size, ways * self.n_support, *x_support.shape[-3:])) # e.g. [1,50*n_support,*(3,84,84)]
        x_query = data[:,:,self.n_support:]
        x_query = x_query.reshape((batch_size, ways * self.n_query, *x_query.shape[-3:])) # e.g. [1,50*n_query,*(3,84,84)]
        
        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(0).unsqueeze(2) # batch and shot dim
        y_query = y_query.repeat(batch_size, 1, self.n_query)
        y_query = y_query.view(batch_size, -1).to(self.device)
        
        y_support = torch.arange(ways).unsqueeze(0).unsqueeze(2) # batch and shot dim
        y_support = y_support.repeat(batch_size, 1, self.n_support)
        y_support = y_support.view(batch_size, -1).to(self.device)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1) # e.g. [1,50*(n_support+n_query),*(3,84,84)]
        z = self.encoder.forward(x)
        z_support = z[:,:ways * self.n_support] # e.g. [1,50*n_support,*(3,84,84)]
        z_query = z[:,ways * self.n_support:] # e.g. [1,50*n_query,*(3,84,84)]

        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        loss, accuracy= prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)
        #choose small loss query examples
        # threshold = sorted(loss_n[0])[-y_query.size(1)//3]#omit 1/3 high loss query examples
        # index=[i for i in range(loss_n.shape[1]) if loss_n[0][i] < threshold]
        # index=[]
        # for x in range(loss_n.shape[1]):
        #     if loss_n[0][x] < threshold:
        #         index.append(x)
        # z_query_spl = z_query[index]
        # y_query_spl = y_query[index]
        return loss, accuracy


class ProtoCLR_fea(nn.Module):
    """Calculate the ProtoCLR loss on a batch of images.

    Parameters:
        - encoder: The encoder network.
        - device: The device to use.
        - n_support: The number of support shots.
        - n_query: The number of query shots.
    """

    def __init__(self, encoder, device, n_support=1, n_query=1,
                 distance='euclidean'):
        super(ProtoCLR_fea, self).__init__()
        self.n_support = n_support
        self.n_query = n_query

        self.encoder = encoder
        self.device = device
        self.distance = distance

    def forward(self, batch):
        # Treat the first dim as way, the second as shots
        # and use the first shot as support (like 1-shot setting)
        # the remaining shots as query
        data = batch['data'].to(self.device)  # [batch_size x ways x shots x image_dim]
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
        x = torch.cat([x_support, x_query], 1)  # e.g. [1,50*(n_support+n_query),*(3,84,84)]
        z = self.encoder.forward(x)
        z_support = z[:, :ways * self.n_support]  # e.g. [1,50*n_support,*(3,84,84)]
        z_query = z[:, ways * self.n_support:]  # e.g. [1,50*n_query,*(3,84,84)]

        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support  # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        # loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
        #                                    distance=self.distance)

        return z


class ProtoCLR_cls(nn.Module):
    """Calculate the ProtoCLR loss on a batch of images.

    Parameters:
        - encoder: The encoder network.
        - device: The device to use.
        - n_support: The number of support shots.
        - n_query: The number of query shots.
    """

    def __init__(self, encoder, device, n_support=1, n_query=1,
                 distance='euclidean'):
        super(ProtoCLR_cls, self).__init__()
        self.n_support = n_support
        self.n_query = n_query

        self.encoder = encoder
        self.device = device
        self.distance = distance

    def forward(self, batch):
        # Treat the first dim as way, the second as shots
        # and use the first shot as support (like 1-shot setting)
        # the remaining shots as query
        data = batch['data'].to(self.device)  # [batch_size x ways x shots x image_dim]
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
        x = torch.cat([x_support, x_query], 1)  # e.g. [1,50*(n_support+n_query),*(3,84,84)]
        z = self.encoder.forward(x)
        z_support = z[:, :ways * self.n_support]  # e.g. [1,50*n_support,*(3,84,84)]
        z_query = z[:, ways * self.n_support:]  # e.g. [1,50*n_query,*(3,84,84)]

        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support  # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        # Calculate loss and accuracies
        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)
        # choose small loss query examples
        # threshold = sorted(loss_n[0])[-y_query.size(1)//3]#omit 1/3 high loss query examples
        # index=[i for i in range(loss_n.shape[1]) if loss_n[0][i] < threshold]
        # index=[]
        # for x in range(loss_n.shape[1]):
        #     if loss_n[0][x] < threshold:
        #         index.append(x)
        # z_query_spl = z_query[index]
        # y_query_spl = y_query[index]
        return loss


class ProtoCLR_dis(nn.Module):
    """Calculate the ProtoCLR loss on a batch of images.

    Parameters:
        - encoder: The encoder network.
        - device: The device to use.
        - n_support: The number of support shots.
        - n_query: The number of query shots.
    """

    def __init__(self, encoder, device, n_support=1, n_query=1,
                 distance='euclidean'):
        super(ProtoCLR_dis, self).__init__()
        self.n_support = n_support
        self.n_query = n_query

        self.encoder = encoder
        self.device = device
        self.distance = distance

        self.fc1 = nn.Linear(256, 50)
        self.fc2 = nn.Linear(50, 1)
    def forward(self, batch):
        e1 = F.relu(self.fc1(batch))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
