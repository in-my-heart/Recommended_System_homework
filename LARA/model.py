import torch
from torch import nn
import torch.utils.data



# 超参数设置
alpha = 0  # 正则项参数
attr_num = 18
attr_present_dim = 5
batch_size = 1024
hidden_dim = 100
user_emb_dim = attr_num
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device =2torch.device('cpu')
epoch = 200


def param_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(m.bias.unsqueeze(0))
    else:
        pass


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(2*attr_num, attr_present_dim)
        self.l1 = nn.Linear(attr_num*attr_present_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.h = nn.Tanh()
        self.__init_param__()

    def __init_param__(self):
        for md in self.G_attr_matrix.modules():
            torch.nn.init.xavier_normal_(md.weight)
        for md in self.modules():
            param_init(md)

    def forward(self, attribute_id):
        attr_present = self.G_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num*attr_present_dim])
        o1 = self.h(self.l1(attr_feature))
        o2 = self.h(self.l2(o1))
        o3 = self.h(self.l3(o2))
        return o3


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(2*attr_num, attr_present_dim)
        self.l1 = nn.Linear(attr_num*attr_present_dim+user_emb_dim, hidden_dim, bias=True)
        self.h = nn.Tanh()
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.__init_param__()

    def __init_param__(self):
        for md in self.D_attr_matrix.modules():
            torch.nn.init.xavier_normal_(md.weight)
        for md in self.modules():
            param_init(md)

    def forward(self, attribute_id, user_emb):
        attribute_id = attribute_id.long()
        attr_present = self.D_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num*attr_present_dim])
        emb = torch.cat((attr_feature, user_emb), 1)
        emb = emb.float()
        o1 = self.h(self.l1(emb))
        o2 = self.h(self.l2(o1))
        d_logit = self.l3(o2)
        d_prob = torch.sigmoid(d_logit)
        return d_prob, d_logit





