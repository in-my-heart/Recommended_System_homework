from model import Generator,Discriminator
import torch
import time
import torch.utils.data
import dataset
import test
def train(g, d, train_loader, neg_loader, epochs, g_optim, d_optim, datasetLen):
    print("train start")
    g = g.to(device)
    d = d.to(device)
    loss = torch.nn.BCELoss()
    start = time.time()
    for i_epo in range(epochs):
        i = 0
        neg_iter = neg_loader.__iter__()
        # 训练D
        d_loss_sum = 0.0
        for user, item, attr, user_emb in train_loader:
            if i*batch_size >= datasetLen:
                break
            # 取出负采样的样本
            _, _, neg_attr, neg_user_emb = neg_iter.next()
            neg_attr = neg_attr.to(device)
            neg_user_emb = neg_user_emb.to(device)
            attr = attr.to(device)
            user_emb = user_emb.to(device)
            fake_user_emb = g(attr)  # 根据item的属性生成用户表达
            d_real, d_logit_real = d(attr, user_emb)
            d_fake, d_logit_fake = d(attr, fake_user_emb)
            d_neg, d_logit_neg = d(neg_attr, neg_user_emb)
            # d_loss分成三部分, 正样本，生成的样本，负样本
            d_optim.zero_grad()
            d_loss_real = loss(d_real, torch.ones_like(d_real))
            d_loss_fake = loss(d_fake, torch.zeros_like(d_fake))
            d_loss_neg = loss(d_neg, torch.zeros_like(d_neg))
            d_loss_sum = torch.mean(d_loss_real + d_loss_fake+d_loss_neg)
            d_loss_sum.backward()
            d_optim.step()
            i += 1
        # 训练G
        g_loss = 0.0
        for user, item, attr, user_emb in train_loader:
            # g loss
            g_optim.zero_grad()
            attr = attr.long()
            attr = attr.to(device)
            fake_user_emb = g(attr)
            fake_user_emb.to(device)
            d_fake, _ = d(attr, fake_user_emb)
            g_loss = loss(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            g_optim.step()
        end = time.time()
        print( 'Epoch: [{0}/{1}]'.format(i_epo, epochs) +
               " time:%.3fs, d_loss:%.3f, g_loss:%.3f " % ((end - start), d_loss_sum, g_loss),end='')
        start = end
        # test---
        item, attr = test.get_test_data()
        item = item.to(device)
        attr = attr.to(device)
        item_user = g(attr)
        test.to_valuate(item, item_user)
        g_optim.zero_grad()  # 生成器清零梯度
        # 保存模型
        if i_epo % 10 == 0:
            torch.save(g.state_dict(), 'data/result/generator_'+str(i_epo)+".pth")
            torch.save(d.state_dict(), 'data/result/discriminator_' + str(i_epo) + ".pth")

# 超参数设置
alpha = 0  # 正则项参数
attr_num = 18
attr_present_dim = 5
batch_size = 1024
hidden_dim = 100
user_emb_dim = attr_num
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
epoch = 100
# 模型训练
train_dataset = dataset.LaraDataset('data/train/train_data.csv', 'data/train/user_emb.csv')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
neg_dataset = dataset.LaraDataset('data/train/neg_data.csv', 'data/train/user_emb.csv')
neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
generator = Generator()
discriminator = Discriminator()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=alpha)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=alpha)
# 因为负样本的数据量要小一些，为了训练方便，使用负样本的长度来训练
train(generator, discriminator, train_loader, neg_loader, epoch, g_optimizer, d_optimizer, neg_dataset.__len__())

# 从test中导入模型进行训练
#import test
#test.load_model_to_test('data/result/g_780.pt')