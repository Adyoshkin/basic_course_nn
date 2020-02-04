import torch
import torch.nn as nn
import torch.utils.data as dt
from carvana_dataset import CarvanaDataset
from model import SegmenterModel
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from tqdm import *
import numpy as np

log = './log/'
train = './data/train/'
train_masks = './data/train_masks/'
test = './data/test/'
test_masks = './data/test_masks'

if os.path.exists(log) == False:
    os.mkdir(log)
tb_writer = SummaryWriter(log_dir='log')

def train_net(useCuda=True, n_epoch = 100):
    """
     Тут модель, которую мы реализовали в файле model.py
    """
    m = SegmenterModel()
    """
    Делаем критерий, который будем оптимайзить
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(m.parameters(), lr=0.001)

    if useCuda == True:
        m = m.cuda()
        criterion= criterion.cuda()

    ds = CarvanaDataset(train, train_masks)
    ds_test = CarvanaDataset(test, test_masks)

    dl      = dt.DataLoader(ds, shuffle=True, num_workers=4, batch_size=5)
    dl_test = dt.DataLoader(ds_test, shuffle=False, num_workers=4, batch_size=5)

    global_iter = 0
    for epoch in range(0, n_epoch):
        print ("Current epoch: ", epoch)
        epoch_loss = 0
        m.train(True)
        for iter, (i, t) in enumerate(tqdm( dl) ):
            i = Variable(i)
            t = Variable(t).long()
            if useCuda :
                i = i.cuda()
                t = t.cuda()
            o = m(i)
            t = t.view((t.shape[0], t.shape[2], t.shape[3]))
            loss = criterion(o, t)
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_loss += loss.data

        epoch_loss = epoch_loss / float(len(ds))
        print ("Epoch loss", epoch_loss)
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)

        print ("Make test")
        test_loss = 0
        m.train(False)

        tb_out = np.random.choice(range(0, len(dl_test)), 3 )
        for iter, (i, t) in enumerate(tqdm(dl_test)):
            i = i.requires_grad_(False)
            t = t.requires_grad_(False).long()
            if useCuda :
                i = i.cuda()
                t = t.cuda()
            o = m(i)
            t = t.view((t.shape[0], t.shape[2], t.shape[3]))
            loss = criterion(o, t)
            o = torch.argmax(o, dim=1)
            test_loss += loss.data

            for k, c in enumerate(tb_out):
                if c == iter:
                    tb_writer.add_image('Image/Test_input_%d'%k,  i[0].cpu(), epoch)  # Tensor
                    tb_writer.add_image('Image/Test_target_%d'%k, t[0].cpu(), epoch, dataformats='HW') # Tensor
                    tb_writer.add_image('Image/Test_output_%d'%k, o[0].cpu(), epoch, dataformats='HW')  # Tensor

        test_loss = test_loss / float(len(ds_test))
        print ("Test loss", test_loss)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)
