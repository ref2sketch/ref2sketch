from model_deep import *
from dataset import *
from perceptual import VGGPerceptualLoss,VGGstyleLoss
import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from statistics import mean
from utils2 import is_image_file, load_img, save_img, test_load_img
import os


class Train_deep:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_l1 = args.wgt_l1
        self.wgt_gan = args.wgt_gan
        self.sparse_loss = args.sparse_loss

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data
        self.direction = args.direction

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, netD=[], optimG=[], optimD=[], epoch=[], mode='train_deep'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train_deep':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG, netD, optimG, optimD, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def preprocess(self, data):
        normalize = Normalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(normalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_l1 = self.wgt_l1
        sparse_loss = self.sparse_loss
        wgt_gan = self.wgt_gan

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
        zeros = np.zeros((1,512,32,32))
        zeros = torch.from_numpy(zeros)
        zeros = zeros.to(device)

        dataset_train = Dataset(dir_data_train, direction=self.direction, data_type=self.data_type, nch=self.nch_in)#, transform=transform_train)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        netG = ResNet(nch_in, nch_out, nch_ker, norm)

        netD = Discriminator(2*nch_in, nch_ker, norm)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_L1 = nn.L1Loss().to(device) 
        fn_GAN = nn.BCEWithLogitsLoss().to(device)
        per_loss = VGGPerceptualLoss().to(device)

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.beta1, 0.999))


        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, netD, optimG, optimD, st_epoch = self.load(dir_chck, netG, netD, optimG, optimD, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            loss_G_l1_train = []
            loss_G_gan_train = []
            loss_G_vgg16 = []

            loss_G_sparse_train = []
            loss_D_real_train = []
            loss_D_fake_train = []
            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = data['dataA'].to(device)
                label = data['dataB'].to(device)
                style = data['dataC'].to(device)

                output,sparse_map= netG(input,style)

                # backward netD
                fake = torch.cat([input, output], dim=1)
                real = torch.cat([input, label], dim=1)

                set_requires_grad(netD, True)
                optimD.zero_grad()

                pred_real = netD(real)
                pred_fake = netD(fake.detach())

                loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                fake = torch.cat([input, output], dim=1)

                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(fake)

                loss_G_gan = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_l1 = fn_L1(output, label)
                per_loss_rec = per_loss(output,label)
                loss_G_sparse = fn_L1(sparse_map, zeros)

                loss_G = (wgt_l1 * loss_G_l1) + (wgt_gan * loss_G_gan) + per_loss_rec  + (loss_G_sparse*sparse_loss)
                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_l1_train += [loss_G_l1.item()]
                loss_G_gan_train += [loss_G_gan.item()]
                loss_G_vgg16 += [per_loss_rec.item()]
                loss_G_sparse_train += [loss_G_sparse.item()]
                loss_D_fake_train += [loss_D_fake.item()]
                loss_D_real_train += [loss_D_real.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'GEN L1: %.4f GEN GAN: %.4f GEN VGG16: %.4f  GEN SPARSE: %.4f  DISC FAKE: %.4f DISC REAL: %.4f'
                      % (epoch, i, num_batch_train,
                         mean(loss_G_l1_train), mean(loss_G_gan_train), mean(loss_G_vgg16),mean(loss_G_sparse_train), mean(loss_D_fake_train), mean(loss_D_real_train)))


            writer_train.add_scalar('loss_G_l1', mean(loss_G_l1_train), epoch)
            writer_train.add_scalar('loss_G_gan', mean(loss_G_gan_train), epoch)
            writer_train.add_scalar('loss_D_fake', mean(loss_D_fake_train), epoch)
            writer_train.add_scalar('loss_D_real', mean(loss_D_real_train), epoch)


            ## save
            if (epoch % num_freq_save) == 0 or epoch == 1:
                net_g_model_out_path = "checkpoints/{}/{}/".format(self.scope,name_data)
                check_point_name = "netG_model_epoch_{}.pth".format(epoch)
                if not os.path.exists(net_g_model_out_path):
                    os.makedirs(net_g_model_out_path)
                torch.save(netG, net_g_model_out_path+check_point_name)


                model_path = "./checkpoints/{}/{}/netG_model_epoch_{}.pth".format(self.scope,name_data, epoch)
                net_g = torch.load(model_path,map_location='cuda:0').to(device)
                if self.direction == "A2B":
                    image_dir = "datasets/{}/test/a/".format(name_data)
                else:
                    image_dir = "datasets/{}/test/b/".format(name_data)

                style_test_image_dir ="datasets/{}/test/c/".format(name_data)

                image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
                style_test_image_dir_name = [x for x in os.listdir(style_test_image_dir) if is_image_file(x)]

                transform_list = [transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]
                                

                transform = transforms.Compose(transform_list)

                img,w,h = test_load_img(image_dir + image_filenames[0])
                style_img = load_img(style_test_image_dir+style_test_image_dir_name[0])

                img = transform(img)
                style_img = transform(style_img)
                
                input = img.unsqueeze(0).to(device)
                style_input = style_img.unsqueeze(0).to(device)
                out,spatial= net_g(input,style_input)
                out_img = out.detach().squeeze(0).cpu()

                if not os.path.exists(os.path.join("result", name_data)):
                    os.makedirs(os.path.join("result", name_data))
                save_img(out_img,w,h ,"result/{}/{}".format(name_data, str(epoch)+image_filenames[0]))

        writer_train.close()

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
