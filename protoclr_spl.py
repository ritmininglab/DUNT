"""Contrastive loss forumulation"""
import torch
import torch.nn as nn

from utils_proto_spl import (prototypical_loss,
                                       get_prototypes)
# def cosine_sim(x,y):
#     breakpoint()
#     res=torch.transpose(x)*y/(torch.norm(x)*torch.norm(y))
#     return res

class ProtoCLR(nn.Module):
    """Calculate the ProtoCLR loss on a batch of images.

    Parameters:
        - encoder: The encoder network.
        - device: The device to use.
        - n_support: The number of support shots.
        - n_query: The number of query shots.
    """
    def __init__(self, encoder, device='cpu', n_support=1, n_query=1,
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
        if len(batch)==1:
            
            data = batch['data'].to(self.device) # [batch_size x ways x shots x image_dim]
            data = data.unsqueeze(0)  # TODO remove when multi-batching is supported
            #data = batch['train'].to(self.device)

            # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
            batch_size = data.size(0)
            ways = data.size(1)
            # breakpoint()
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
        else:
            # breakpoint()
            data = batch['train'][0].to(self.device)
            ways = data.size(1)
            x_support = batch["train"][0].to(self.device)
            x_query = batch["test"][0].to(self.device)
            y_support = batch["train"][1].to(self.device)
            y_query = batch["test"][1].to(self.device)

            # x = torch.cat([x_support, x_query], 1)
            # z = self.encoder.forward(x)
            # shots = int(x_support.size(1) / ways)
            # test_shots = int(x_query.size(1) / ways)
            # z_support = z[:, :ways * shots]
            # z_query = z[:, ways * shots:]

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
        #compute similarity between supports
        _,_,_,predictions_sup = prototypical_loss(z_proto, z_support, y_support,distance=self.distance)
        cos_res=0
        for i in range(predictions_sup.shape[1]-1):
            for j in range(i,1,predictions_sup.shape[1]-1):
                x= predictions_sup[0,i,:]
                y=predictions_sup[0,j+1,:]
                # breakpoint()
                cos_res+=torch.sum(torch.mul(x,y))/(torch.norm(x)*torch.norm(y))

        # Calculate loss and accuracies
        loss, accuracy,loss_n,predictions = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)  

        return z,loss, accuracy,predictions,cos_res



class ProtoCLR_spl(nn.Module):
    """Calculate the ProtoCLR loss on a batch of images.

    Parameters:
        - encoder: The encoder network.
        - device: The device to use.
        - n_support: The number of support shots.
        - n_query: The number of query shots.
    """

    def __init__(self, encoder, device='cpu', n_support=1, n_query=1,
                 distance='euclidean',spl=3):
        super(ProtoCLR_spl, self).__init__()
        self.n_support = n_support
        self.n_query = n_query

        self.encoder = encoder
        self.device = device
        self.distance = distance
        self.spl=spl

    def forward(self, batch):
        # Treat the first dim as way, the second as shots
        # and use the first shot as support (like 1-shot setting)
        # the remaining shots as query
        if len(batch)==1:
            data = batch['data'].to(self.device) # [batch_size x ways x shots x image_dim]
          # [batch_size x ways x shots x image_dim]
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
        else:
            data = batch['train'][0].to(self.device)
            ways = data.size(1)
            x_support = batch["train"][0].to(self.device)
            x_query = batch["test"][0].to(self.device)
            y_support = batch["train"][1].to(self.device)
            y_query = batch["test"][1].to(self.device)
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
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query, y_query,
                                                   distance=self.distance)
        # choose small loss query examples
        threshold = sorted(loss_n[0])[-int(y_query.size(1) * self.spl)]  # omit 1/3 high loss query examples
        index = [i for i in range(loss_n.shape[1]) if loss_n[0][i] < threshold+0.001]
        y_query_spl = y_query[:, index]
        z_query_spl = z_query[:, index, :]
        loss, accuracy, loss_n,predictions = prototypical_loss(z_proto, z_query_spl, y_query_spl,
                                                   distance=self.distance)
        del loss_n

        return z,loss, accuracy,predictions,index



