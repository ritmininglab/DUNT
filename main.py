import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR

import numpy as np
import copy
import scipy.stats as st
from tqdm import tqdm
import shutil
import os
import pickle
from prototransfer.protoclr_spl import ProtoCLR,ProtoCLR_spl
from prototransfer.protonet_spl import Protonet,Protonet_spl
from prototransfer.models import CNN_4Layer, ResNet18

from prototransfer.unlabelled_loader import UnlabelledDataset
from prototransfer.supervised_finetuning import supervised_finetuning
from prototransfer.episodic_loader import get_episode_loader
from torch.autograd import Variable
from prototransfer.protoaux import Protoaux,Protoaux_fea,Protoaux_dis
from torch.autograd import grad
# from signal import signal, SIGPIPE, SIG_DFL
# #Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
# signal(SIGPIPE,SIG_DFL)

def main(args, mode='train'):
    ##########################################
    # Setup scp -r D:\\ProtoTransfer-master xq5054@yu-r1.gccis.rit.edu:/shared/users/xq5054
    ##########################################

    # Check whether GPU is available
    cuda = torch.cuda.is_available()
    cuda_device = args.cuda_device

    device = torch.device('cuda:{cuda_device}'.format(cuda_device=cuda_device) if cuda else 'cpu')
    # device = torch.device('cuda' if cuda else 'cpu')
    print('Used device:', device)

    # Define datasets and loaders
    num_workers = args.num_data_workers_cuda if cuda else args.num_data_workers_cpu
    kwargs = {'num_workers': num_workers}
    plot_history = {
        'train_loss': [],
        'train_acc': []
    }
    # Load data for training
    if (mode == 'train') or (mode == 'trainval'):
        dataset_train = UnlabelledDataset(args.dataset,
                                          args.datapath, split='train',
                                          transform=None,
                                          n_images=args.n_images,
                                          n_classes=args.n_classes,
                                          n_support=args.train_support_shots,
                                          n_query=args.train_query_shots,
                                          no_aug_support=args.no_aug_support,
                                          no_aug_query=args.no_aug_query)

        # Optionally add validation set to training
        if args.merge_train_val:
            dataset_val = UnlabelledDataset(args.dataset, args.datapath, 'val',
                                            transform=None,
                                            n_support=args.train_support_shots,
                                            n_query=args.train_query_shots,
                                            no_aug_support=args.no_aug_support,
                                            no_aug_query=args.no_aug_query)

            dataset_train = ConcatDataset([dataset_train, dataset_val])

        # Train data loader
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=torch.cuda.is_available())

        if mode == 'trainval': # USe training set for ProtoCLR evaluation
            dataloader_test = dataloader_train

    # Test on the validation set
    elif mode == 'val': 
        dataloader_test = get_episode_loader(args.dataset, args.datapath,
                                             ways=args.eval_ways,
                                             shots=args.eval_support_shots,
                                             test_shots=args.eval_query_shots,
                                             batch_size=args.batch_size,
                                             split='val',
                                             **kwargs)
        # The rest is identical to mode == test
        mode = 'test'

    # Load data for testing
    elif mode == 'test': 
        # dataloader_test = get_episode_loader(args.dataset, args.datapath,
        #                                      ways=args.eval_ways,
        #                                      shots=args.eval_support_shots,
        #                                      test_shots=args.eval_query_shots,
        #                                      batch_size=args.batch_size,
        #                                      split='test',
        #                                      **kwargs)
        dataloader_test1 = get_episode_loader(args.dataset, args.datapath,
                                              ways=args.eval_ways,
                                              shots=1,
                                              test_shots=args.eval_query_shots,
                                              batch_size=args.batch_size,
                                              split='test',
                                              **kwargs)
        dataloader_test5 = get_episode_loader(args.dataset, args.datapath,
                                              ways=args.eval_ways,
                                              shots=5,
                                              test_shots=args.eval_query_shots,
                                              batch_size=args.batch_size,
                                              split='test',
                                              **kwargs)
    elif mode == 'train_ssl':
        dataloader_pri = get_episode_loader(args.dataset, args.datapath,
                           ways=args.train_ways,
                           shots=args.train_support_shots,
                           test_shots=args.train_query_shots,
                           batch_size=args.batch_size,
                           split='train',
                           **kwargs)
        dataset_aux = UnlabelledDataset(args.dataset,
                                          args.datapath, split='train',
                                          transform=None,
                                          n_images=args.n_images,
                                          n_classes=args.n_classes,
                                          n_support=args.train_support_shots,
                                          n_query=args.train_query_shots,
                                          no_aug_support=args.no_aug_support,
                                          no_aug_query=args.no_aug_query)


        dataloader_aux = DataLoader(dataset_aux,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=torch.cuda.is_available())
        dataloader_test1 = get_episode_loader(args.dataset, args.datapath,
                                             ways=args.eval_ways,
                                             shots=1,
                                             test_shots=args.eval_query_shots,
                                             batch_size=args.batch_size,
                                             split='test',
                                             **kwargs)
        dataloader_test5 = get_episode_loader(args.dataset, args.datapath,
                                             ways=args.eval_ways,
                                             shots=5,
                                             test_shots=args.eval_query_shots,
                                             batch_size=args.batch_size,
                                             split='test',
                                             **kwargs)
    elif mode == 'test_un':
        dataset_aux = UnlabelledDataset(args.dataset,
                                          args.datapath, split='train',
                                          transform=None,
                                          n_images=args.n_images,
                                          n_classes=args.n_classes,
                                          n_support=args.train_support_shots,
                                          n_query=args.train_query_shots,
                                          no_aug_support=args.no_aug_support,
                                          no_aug_query=args.no_aug_query)


        dataloader_test1 = DataLoader(dataset_aux,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=torch.cuda.is_available())
        dataloader_test5 = DataLoader(dataset_aux,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=torch.cuda.is_available())

    # Define model
    ## Encoder
    if args.dataset in ['omniglot', 'doublemnist', 'triplemnist']:
        channels = 1
    elif args.dataset in ['miniimagenet','miniimagenet_noisy', 'tieredimagenet', 'cub','cub_noisy', 'cifar_fs','eurosat','isic','cropdisease','chestx']:
        channels = 3
    else:
        raise ValueError('No such dataset')
    if (args.backbone == 'conv4') or (args.backbone == 'cnn'):
        encoder = CNN_4Layer(in_channels=channels)
    elif(args.backbone == 'res18'):
        encoder = ResNet18()
    else:
        raise ValueError('No such model')
    encoder = encoder.to(device)

    ## Protonet + Loss
    if args.pri:
        proto = Protonet_spl(encoder, distance=args.distance, device=device, spl=args.spl)
    else:
        proto = Protonet(encoder, distance=args.distance, device=device)
    proto_mixup = Protonet(encoder, distance=args.distance, device=device)
    if (mode == 'train') or (mode == 'trainval') or  (mode == 'train_ssl') :
        if args.aux:
            self_sup_loss = ProtoCLR_spl(encoder, n_support=args.train_support_shots,
                                 n_query=args.train_query_shots,
                                 device=device, distance=args.distance, spl=args.spl)
        else:
            self_sup_loss = ProtoCLR(encoder, n_support=args.train_support_shots,
                                         n_query=args.train_query_shots,
                                         device=device, distance=args.distance)
    protoaux_dis = Protoaux_dis(encoder, distance=args.distance, device=device,dataset=args.dataset)
    if mode == 'train' or mode == 'train_ssl':
        # Define optimisation parameters
        if args.mixup:
            optimizer = Adam(
                list(self_sup_loss.parameters()) + list(proto.parameters()) + list(proto_mixup.parameters()),
                lr=args.lr)
            scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

        else:
            # optimizer = Adam(
            #     list(self_sup_loss.parameters()) + list(proto.parameters()),
            #     lr=args.lr)
            optimizer = Adam(list(proto.parameters()) + list(self_sup_loss.parameters()), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
            optimizer_dis = Adam(protoaux_dis.parameters(), lr=args.lr)
            scheduler_dis = StepLR(optimizer_dis, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)


        # Save path
        os.makedirs('prototransfer/checkpoints', exist_ok=True)
        os.makedirs('plot', exist_ok=True)
        plot_path = os.path.join('plot/plot_history_OTSPLCOS{}_{}_{}_noisy_{}_{}_{}supp_{}query_{}bs_{}alpha_{}remove_{}lr' \
                                 .format(args.spl,args.self_sup_loss,
                                         args.dataset, args.backbone,
                                         args.distance,
                                         args.train_support_shots,
                                         args.train_query_shots,
                                         args.batch_size,
                                         args.alpha,args.spl,args.lr))
        if args.noise:
            default_path = os.path.join('checkpoints/OTSPLCOSnoise_{}_{}_{}supp_{}query_{}bs_{}iter_spl{}_pri{}_aux{}_{}alpha_{}beta_{}epochs_{}lr'\
                                    .format(args.dataset, args.dataset_aux,
                                            args.train_support_shots,args.train_query_shots,
                                            args.batch_size,args.iterations,args.spl,args.pri,args.aux,args.alpha,args.beta,args.epochs,args.lr))
        else:
            default_path = os.path.join('checkpoints/OTSPLCosnoise_{}_{}_{}supp_{}query_{}bs_{}iter_spl{}_pri{}_aux{}_{}alpha_{}beta_{}epochs_{}lr'\
                                    .format(args.dataset, args.dataset_aux,
                                            args.train_support_shots,args.train_query_shots,
                                            args.batch_size,args.iterations,args.spl,args.pri,args.aux,args.alpha,args.beta,args.epochs,args.lr))
        if args.n_images is not None:
            default_path += '_' + str(args.n_images) + 'images'
        if args.n_classes is not None:
            default_path += '_' + str(args.n_classes) + 'classes'
        save_path = args.save_path or default_path
        print('Save path is:', save_path)

        def save_checkpoint(state, is_best):
            if args.save:
                filename = save_path + '.pth.tar'
                torch.save(state, filename)
                if is_best:
                    shutil.copyfile(filename, save_path + '_best.pth.tar')

        # Load path
        if args.load_last:
            args.load_path = default_path + '.pth.tar'
        if args.load_best:
            args.load_path = default_path + '_best.pth.tar'

        # Load training state
        n_no_improvement = 0
        best_loss = np.inf
        best_accuracy = 0
        start_epoch = 0

        # Adjust patience
        if args.patience < 0:
            print('No early stopping!')
            args.patience = np.inf
        else:
            print('Early stopping with patience {} epochs'.format(args.patience))

    # Load checkpoint
    if args.load_path:
        try: # Cannot load CUDA trained models onto cpu directly
            checkpoint = torch.load(args.load_path)
        except:
            checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'))
        proto.encoder.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        if mode == 'train' or mode == 'train_ssl':
            n_no_improvement = checkpoint['n_no_improvement']
            best_loss = checkpoint['best_loss']
            best_accuracy = checkpoint['best_accuracy']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
            scheduler_dis.load_state_dict(checkpoint['scheduler_dis'])
        print("Loaded checkpoint '{}' (epoch {})"
              .format(args.load_path, start_epoch))

    def gradient_penalty(critic, h_s, h_t, device=None):
        ''' Gradeitnt penalty approach'''
        alpha = torch.rand(h_s.size(0), 1).cuda().to(device)
        differences = h_t - h_s
        # try:
        #     differences = h_t - h_s
        # except:
        #     print('h_t', h_t.size())
        #     print('h_s',h_s.size())
        
        interpolates = h_s + (alpha * differences)
        interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
        h_s.requires_grad_()
        preds = critic.loss(h_s)
        gradients = grad(preds, h_s,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty
    def set_requires_grad(model, requires_grad=True):
        """
        Used in training adversarial approach
        :param model:
        :param requires_grad:
        :return:
        """

        for param in model.parameters():
            param.requires_grad = requires_grad
    ##########################################
    # Define train and test functions
    ##########################################

    def train_epoch(model,proto,protoaux_dis,dataloader_pri, dataloader_aux, optimizer, scheduler, optimizer_dis, scheduler_dis):
        model.train()
        proto.train()
        protoaux_dis.train()
        accuracies_pri = []
        accuracies_aux = []
        losses_pri = []
        losses_aux = []
        # episodes = enumerate(dataloader_pri)
        
        iter = 0
        for episode, batch in zip(dataloader_pri, dataloader_aux):
            iter += 0
            optimizer.zero_grad()
            n_aux = 0
            cos_batch = []
            batch_ = []
            # remove the confusing tasks
            for i,batch in enumerate(dataloader_aux):
                batch_.append(batch)
                aux_z,loss_aux, accuracy_aux,_,cos_res = model.forward(batch)
                cos_batch.append(cos_res.detach().cpu().numpy())
                
                    # break
            batch_num = np.argmin(cos_batch)
            cos_min = np.min(cos_batch)
            # breakpoint()
            batch = batch_[batch_num]
            pri_z,loss_pri, accuracy,_,_ = proto.loss(episode, args.train_ways)
            aux_z,loss_aux, accuracy_aux,_,cos_res = model.forward(batch)
            
            accuracies_pri.append(accuracy)
            accuracies_aux.append(accuracy_aux)
            losses_pri.append(loss_pri.item())
            total_loss = 2*args.alpha*loss_pri + 2*(1-args.alpha)*loss_aux

            # total_loss = 2*args.alpha*loss_pri.clone() + 2*(1-args.alpha)*loss_aux.clone()
            #total_loss = loss_pri.clone() + loss_aux.clone()
            gamma_ratio = 1  # 0.1
            wassertein_distance = protoaux_dis.loss(aux_z).mean() - gamma_ratio * protoaux_dis.loss(pri_z).mean()

            if args.dataset=='cub' or args.dataset=='tieredimagenet':
                # with torch.no_grad():
                #     pri_z, _, _, _, _ = proto.loss(episode, args.train_ways)
                #     aux_z, _, _, _, _ = model.forward(batch)
                # gp = gradient_penalty(protoaux_dis, aux_z,pri_z)
                # print('gp-value1',gp)
                gp = torch.Tensor([0.982]).to(device)

            else:
                with torch.no_grad():
                    pri_z, _, _, _, _ = proto.loss(episode, args.train_ways)
                    aux_z, _, _, _, _ = model.forward(batch)
                gp = gradient_penalty(protoaux_dis, aux_z,pri_z, device=device)
                # print('gp-value1',gp)
            # alpha_2= args.beta

            loss = total_loss + args.beta * wassertein_distance + args.beta * gp
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pri_z, _, _, _, _ = proto.loss(episode, args.train_ways)
                aux_z, _, _, _, _ = model.forward(batch)
            for _ in range(1):
                # gradient ascent for multiple times like GANS training

                if args.dataset == 'cub' or args.dataset=='tieredimagenet':
                    # gp = gradient_penalty(protoaux_dis, aux_z, pri_z)
                    # print('gp-value2', gp)

                    gp = torch.Tensor([0.9828]).to(device)

                else:
                    gp = gradient_penalty(protoaux_dis, aux_z, pri_z, device=device)

                wassertein_distance = protoaux_dis.loss(aux_z).mean() - gamma_ratio * protoaux_dis.loss(pri_z).mean()

                dis_loss = -1 * args.beta * wassertein_distance - args.beta * gp * 2

                optimizer_dis.zero_grad()
                dis_loss.backward()
                optimizer_dis.step()
                scheduler_dis.step()
            if iter == args.iterations - 1:
                break
        plot_history['train_loss'].append(losses_pri)
        plot_history['train_acc'].append(accuracies_pri)
        f = open(plot_path, 'wb')
        pickle.dump(plot_history, f)
        # print('save plot')
        return np.mean(losses_pri), np.mean(accuracies_pri),accuracies_pri,accuracies_aux


    def eval_epoch(model, dataloader, iterations,
                   sup_finetune=False, progress_bar=False,
                   trainval=False):
        model.eval()
        losses = []
        accuracies = []
        # Keep original state to reset after finetuning (deepcopy required)
        if not trainval:
            original_encoder_state = copy.deepcopy(model.encoder.state_dict())

        # Define iterator
        episodes = enumerate(dataloader)
        if progress_bar:
            episodes = tqdm(episodes, unit='episodes',
                            total=iterations, initial=0, leave=False)

        # Perform evaulation episodes
        for iteration, episode in episodes:
            if sup_finetune:
                loss, accuracy = supervised_finetuning(model.encoder,
                                                        episode=episode,
                                                        inner_lr=args.sup_finetune_lr,
                                                        total_epoch=args.sup_finetune_epochs,
                                                        freeze_backbone=args.ft_freeze_backbone,
                                                        finetune_batch_norm=args.finetune_batch_norm,
                                                        device=device,
                                                        n_way=args.eval_ways)
                model.encoder.load_state_dict(original_encoder_state)
            elif trainval: # evaluating ProtoCLR loss/accuracy on train set
                with torch.no_grad():
                    loss, accuracy = model.forward(episode)
            else: 
                with torch.no_grad():
                    _, loss, accuracy, _, _ = model.loss(episode, args.eval_ways)

            losses.append(loss.item())
            accuracies.append(accuracy)

            if iteration == iterations - 1:
                break

        conf_interval = st.t.interval(0.95, len(accuracies)-1, loc=np.mean(accuracies),
                                      scale=st.sem(accuracies))
        return np.mean(losses), np.mean(accuracies), np.std(accuracies), conf_interval

    def train(self_sup_loss, proto, trainloader_pri,trainloader_aux,# valloader,
              optimizer, scheduler,
              n_no_improvement=0, best_loss=np.inf, best_accuracy=0,
              start_epoch=0):
        epochs = tqdm(range(start_epoch, args.epochs), unit='epochs',
                      total=args.epochs, initial=start_epoch)
        for epoch in epochs:
            # Train
            loss, accuracy,accuracies_pri,accuracies_aux = train_epoch(self_sup_loss,proto,protoaux_dis,dataloader_pri, dataloader_aux, optimizer, scheduler, optimizer_dis, scheduler_dis)
            # Validation
            # Validation
            loss_test, accuracy_test, _, _ = eval_epoch(proto, dataloader_test1,100)
            loss_test5, accuracy_test5, _, _ = eval_epoch(proto, dataloader_test5, 100)

            # Record best model, loss, accuracy
            best_epoch = accuracy > best_accuracy
            #best_epoch = loss < best_loss
            if best_epoch or epoch % 20 == 0:
                best_loss = loss
                best_accuracy = accuracy

                # Save checkpoint
                save_checkpoint({
                    'epoch': epoch + 1,
                    'n_no_improvement': n_no_improvement,
                    'model': proto.encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': loss,
                    'best_loss': best_loss,
                    'best_accuracy': best_accuracy,
                    'accuracy': accuracy,
                    'setup': args,
                    'accuracies_pri': accuracies_pri,
                    'accuracies_aux': accuracies_aux,
                    'protoaux_dis': protoaux_dis.encoder.state_dict(),
                    'optimizer_dis': optimizer_dis.state_dict(),
                    'scheduler_dis': scheduler_dis.state_dict(),
                    }, best_epoch)

                n_no_improvement = 0

            else:
                n_no_improvement += 1

            # Update progress bar information
            epochs.set_description_str(desc='Epoch {}, loss {:.4f}, accuracy {:.4f}, accuracy-test {:.4f}, accuracy-test5 {:.4f}'\
                                       .format(epoch+1, loss, accuracy,accuracy_test,accuracy_test5))

            # Early stopping
            if n_no_improvement > args.patience:
                print('Early stopping at epoch {}, because there was no\
                      improvement for {} epochs'.format(epoch, args.patience))
                break
        test_loss, test_accuracy, _, test_conf_interval = eval_epoch(proto, dataloader_test1, args.iterations)
        # loss_best, accuracy_best, _, test_conf_interval_best = eval_epoch(best_model, dataloader_test1, args.iterations)
        print('Test loss 1-shot {:.4f} and accuracy 1-shot {:.2f} +- {:.2f}'.format(test_loss,
                                                                                    test_accuracy * 100,
                                                                                    (test_conf_interval[
                                                                                         1] - test_accuracy) * 100))

        test_loss, test_accuracy, _, test_conf_interval = eval_epoch(proto, dataloader_test5, args.iterations)
        # loss_best, accuracy_best, _, test_conf_interval_best = eval_epoch(best_model, dataloader_test5, args.iterations)
        print('Test loss 5-shot {:.4f} and accuracy 5-shot {:.2f} +- {:.2f}'.format(test_loss,
                                                                                    test_accuracy * 100,
                                                                                    (test_conf_interval[
                                                                                         1] - test_accuracy) * 100))
        print('-------------- Best validation loss {:.4f} with accuracy {:.4f}'.format(best_loss, best_accuracy))

        best_checkpoint = torch.load(save_path + '_best.pth.tar')
        best_model = best_checkpoint['model']
        proto.encoder.load_state_dict(best_model)
        return proto.encoder.load_state_dict(best_model)

    ##########################################
    # Run training and evaluation
    ##########################################

    # Training
    if mode == 'train' or mode == 'train_ssl':
        print('Setting:')
        print(args, '\n')
        print('Training ...')
        model = train(self_sup_loss, proto, dataloader_pri, dataloader_aux,  # dataloader_val,
                           optimizer, scheduler, n_no_improvement,
                           best_loss, best_accuracy, start_epoch)
       


    elif mode == 'trainval':
        best_model = self_sup_loss
    else:
        best_model = proto

    # Evaluation
    if (mode == 'test') or (mode == 'trainval') or (mode == 'test_un'):
        print('Evaluating ' + args.load_path + '...')
        test_loss1, test_accuracy1, test_accuracy_std1, test_conf_interval1 \
                = eval_epoch(best_model, dataloader_test1, args.test_iterations,
                             progress_bar=True, sup_finetune=args.sup_finetune,
                             trainval=mode=='trainval')

        print('Test loss1 {:.4f} and accuracy {:.2f} +- {:.2f}'.format(test_loss1,
                                                                      test_accuracy1*100,
                                                                      (test_conf_interval1[1]-test_accuracy1)*100))
        test_loss5, test_accuracy5, test_accuracy_std5, test_conf_interval5 \
            = eval_epoch(best_model, dataloader_test5, args.test_iterations,
                         progress_bar=True, sup_finetune=args.sup_finetune,
                         trainval=mode == 'trainval')

        print('Test loss5 {:.4f} and accuracy5 {:.2f} +- {:.2f}'.format(test_loss5,
                                                                       test_accuracy5 * 100,
                                                                       (test_conf_interval5[1] - test_accuracy5) * 100))

