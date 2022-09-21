import numpy as np
import torch
import skimage
import torchvision.transforms as transforms

#from skimage import transform
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from tps_transformation import tps_transform

class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, mode='train', direction='A2B', data_type='float32', nch=1):#, transform=[]):
        self.data_dir = data_dir

        #self.transform = transform
        self.direction = direction
        self.data_type = data_type
        self.nch = nch
        self.mode = mode

        lst_data_A = os.listdir(data_dir+'/a')
        lst_data_B = os.listdir(data_dir+'/b')
        lst_data_C = os.listdir(data_dir+'/c')

        self.names_A = lst_data_A
        self.names_B = lst_data_B
        self.names_C = lst_data_C

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          #transforms.Normalize([0.5], [0.5]),
                          ]

        self.transform = transforms.Compose(transform_list)


    def __getitem__(self, index):
      
        data_A = Image.open(os.path.join(self.data_dir+'/a', self.names_A[index])).convert('RGB')
        data_B = Image.open(os.path.join(self.data_dir+'/b', self.names_B[index])).convert('L')
        style = Image.open(os.path.join(self.data_dir+'/b', self.names_B[index])).convert('L')
        if random.random() <0.5:
          style = transforms.RandomRotation(90,expand=False)(style)

        style = tps_transform(np.array(style))
        style = Image.fromarray(style)


        #style.save('./aug.png','png')
        #style.save('./aug/'+self.names_A[index],'png')
        
        #data_B = plt.imread(os.path.join(self.data_dir+'/c', self.names_B[index]))#[:, :, :self.nch]

        #if data.dtype == np.uint8:
        #    data = data / 255.0
        #sz = int(data.shape[1]/2)
        trainsize = 286
        data_A = data_A.resize((trainsize, trainsize), Image.BICUBIC)
        data_B = data_B.resize((trainsize, trainsize), Image.BICUBIC)
        style = style.resize((trainsize, trainsize), Image.BICUBIC)

        data_A = transforms.ToTensor()(data_A)
        data_B = transforms.ToTensor()(data_B)
        style = transforms.ToTensor()(style)
        if random.random() <0.5:
          data_A = transforms.ColorJitter(brightness=(0.2, 2),contrast=(0.3, 2),saturation=(0.2, 2),hue=(-0.3, 0.3))(data_A)
        data_A = transforms.Grayscale()(data_A)
        if random.random() <0.5:
          style = torch.fliplr(style)

        if random.random() <0.3:
          data_A = torch.flipud(data_A)
          data_B = torch.flipud(data_B)
          style = torch.flipud(style)

        #data_A=np.array(data_A)
        #data_B=np.array(data_B)
        w_offset = random.randint(0, max(0, trainsize - 256 - 1))
        h_offset = random.randint(0, max(0, trainsize - 256 - 1))
        data_A = data_A[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        data_B = data_B[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        style = style[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        data_A = transforms.Normalize([0.5], [0.5])(data_A)
        data_B = transforms.Normalize([0.5], [0.5])(data_B)
        style = transforms.Normalize([0.5], [0.5])(style)

        if random.random() < 0.5:
            idx = [i for i in range(data_A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            data_A = data_A.index_select(2, idx)
            data_B = data_B.index_select(2, idx)
            style = style.index_select(2,idx)

        if self.mode =='test':
            style = plt.imread(os.path.join(self.data_dir+'/c', self.names_C[index]))#[:, :, :self.nch]

        if self.direction == 'A2B':
            data = {'dataA': data_A, 'dataB': data_B, 'dataC':style}
        else:
            data = {'dataA': data_B, 'dataB': data_A,  'dataC':style}

        #if self.transform:
        #    data = self.transform(data)

        return data

    def __len__(self):
        return len(self.names_A)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data
        dataA, dataB, dataC = data['dataA'], data['dataB'], data['dataC']
        dataA = dataA[:,:,np.newaxis]
        dataB = dataB[:,:,np.newaxis]
        dataC = dataC[:,:,np.newaxis]

        #dataA = np.repeat(dataA,3,axis=2)
        #print(dataA.shape)
        #dataC = dataC[:,:,np.newaxis]
        #dataC = np.repeat(dataC,3,axis=2)

        dataA = dataA.transpose((2, 0, 1)).astype(np.float32)
        dataB = dataB.transpose((2, 0, 1)).astype(np.float32)
        dataC = dataC.transpose((2, 0, 1)).astype(np.float32)

        return {'dataA': torch.from_numpy(dataA), 'dataB': torch.from_numpy(dataB), 'dataC': torch.from_numpy(dataC)}


class Normalize(object):
    def __call__(self, data):
        # Nomalize [0, 1] => [-1, 1]

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data

        dataA, dataB, dataC = data['dataA'], data['dataB'], data['dataC']
        dataA = 2 * dataA - 1
        dataB = 2 * dataB - 1
        dataC = 2 * dataC - 1

        return {'dataA': dataA, 'dataB': dataB, 'dataC':dataC}


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        dataA, dataB, dataC = data['dataA'], data['dataB'], data['dataC']

        if np.random.rand() > 0.5:
            #dataA = np.fliplr(dataA)
            #dataB = np.fliplr(dataB)
            dataC = np.fliplr(dataC)

        if np.random.rand() > 0.5:
            dataA = np.flipud(dataA)
            dataB = np.flipud(dataB)
            dataC = np.fliplr(dataC)

        return {'dataA': dataA, 'dataB': dataB, 'dataC':dataC}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    dataA, dataB ,dataC= data['dataA'], data['dataB'],data['dataC']

    h, w = dataA.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    # dataA = transform.resize(dataA, (new_h, new_w))
    # dataB = transform.resize(dataB, (new_h, new_w))
    # dataC = transform.resize(dataC, (new_h, new_w))

    return {'dataA': dataA, 'dataB': dataB, 'dataC':dataC}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    dataA, dataB ,dataC= data['dataA'], data['dataB'],data['dataC']

    h, w = dataA.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    dataA = dataA[top: top + new_h, left: left + new_w]
    dataB = dataB[top: top + new_h, left: left + new_w]
    dataC = dataC[top: top + new_h, left: left + new_w]

    return {'dataA': dataA, 'dataB': dataB, 'dataC':dataC}


class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}


class Denomalize(object):
    def __call__(self, data):
        # Denomalize [-1, 1] => [0, 1]

        # for key, value in data:
        #     data[key] = (value + 1) / 2 * 255
        #
        # return data

        return (data + 1) / 2

        # input, label = data['input'], data['label']
        # input = (input + 1) / 2 * 255
        # label = (label + 1) / 2 * 255
        # return {'input': input, 'label': label}
