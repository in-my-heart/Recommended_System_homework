import numpy as np
import matplotlib.pyplot as plt
class PMF():
    def __init__(self, feature_vector=64, lr=0.001, lambda_reg_user=0.1, lambda_reg_item=0.1,epoch=100,user_num=0, item_num=0 ):
        self.feature_vector = feature_vector
        self.lr = lr
        self.lambda_reg_user = lambda_reg_user  #  regularization
        self.lambda_reg_item = lambda_reg_item
        self.epoch = epoch
        self.user_num = user_num
        self.item_num = item_num
        self.rmse_train=[]
        self.rmse_test=[]

    def train(self,train,test):
        U=np.random.normal(0,0.1,(num_user,self.feature_vector))
        V = np.random.normal(0, 0.1, (num_item, self.feature_vector))

        for iter in range(self.epoch):
            loss=0.0
            for data in train:
                user=data[0]
                item=data[1]
                rating=data[2]
                predict_rating=np.dot(U[user],V[item].T)
                error=rating-predict_rating
                loss+=error**2
                U[user]+=self.lr*(error*V[item]-self.lambda_reg_user*U[user])
                V[item]+=self.lr*(error*U[user]-self.lambda_reg_item*V[item])
                # U[user]+=self.lr*(error*V[item])
                # V[item]+=self.lr*(error*U[user])
                loss+=self.lambda_reg_user*np.square(U[user]).sum()+self.lambda_reg_item*np.square(V[item]).sum()
            loss=0.5*loss
            _train_rmse=self.eval_rmse(U, V, train)
            _test_rmse=self.eval_rmse(U, V, test)
            self.rmse_train.append(_train_rmse)
            self.rmse_test.append(_test_rmse)
            print('epoch:%d loss:%.3f train_rmse:%.3f test_rmse:%.3f'%(iter,loss,_train_rmse,_test_rmse))

    def eval_rmse(self,U,V,test):
        test_count=len(test)
        tmp_rmse=0.0
        # user = test[0][0]
        # item = test[0][1]
        # real_rating = test[0][2]
        # print(real_rating)
        # predict_rating = np.dot(U[user], V[item].T)
        # print(predict_rating)
        # tmp_rmse += np.square(real_rating - predict_rating)
        for te in test:
            user=te[0]
            item=te[1]
            real_rating=te[2]
            predict_rating=np.dot(U[user],V[item].T)
            tmp_rmse+=np.square(real_rating-predict_rating)
        rmse=np.sqrt(tmp_rmse/test_count)
        return rmse

def read_data(path,ratio):
    user_set={}
    item_set={}
    u_idx=0
    i_idx=0
    data=[]
    with open(path) as f:
        #get num
        for line in f.readlines():
            u,i,r,_= line.split('\t')
            if u not in user_set:
                user_set[u]=u_idx
                u_idx+=1
            if i not in item_set:
                item_set[i]=i_idx
                i_idx+=1
            data.append([user_set[u],item_set[i],float(r)])

    np.random.shuffle(data)
    train=data[0:int(len(data)*ratio)]
    test=data[int(len(data)*ratio):]
    return u_idx,i_idx,train,test
def plot(pmf):
    plt.plot(range(pmf.epoch), pmf.rmse_train, marker='o', label='training')
    plt.plot(range(pmf.epoch), pmf.rmse_test, marker='v', label='test')
    plt.title('movieslen')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
if __name__=='__main__':
    num_user,num_item,train,test=read_data('data/ml-100k/u.data',0.8)
    pmf=PMF(user_num=num_user,item_num=num_item)
    pmf.train(train,test)
    plot(pmf)
