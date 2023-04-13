
import torch
import torch.nn as nn


# Matrix Factorization Model
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # initializing weights
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        
    def forward(self, users, movies):
        U = self.user_emb(users)
        V = self.item_emb(movies)
        return (U*V).sum(1)


class MF_bias(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF_bias, self).__init__()

        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        # initializing weights
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
        
    def forward(self, users, movies):
        U = self.user_emb(users)
        V = self.item_emb(movies)
        b_u = self.user_bias(users).squeeze()
        b_v = self.item_bias(movies).squeeze()
        return (U*V).sum(1) + b_u + b_v
    


def train_epochs(model, train_data, test_data=None, epochs=10, device='cpu', lr=0.01, wd=0.0, save=False, verbose=False):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()

    history = {}
    history['train_loss'] = []
    history['valid_loss'] = []

    for epoch in range(1, epochs + 1):
        model.train()

        '''
        CSR Matrix ==========================
        rows, cols = train_data.nonzero()
        users = torch.LongTensor(rows).to(device) #.cuda()
        items = torch.LongTensor(cols).to(device) #.cuda()
        ratings = torch.FloatTensor(np.array(train_data[rows, cols]).squeeze()).to(device) #.cuda()
        '''
        
        # RATING DataFrame ==================
        users = torch.LongTensor(train_data.userId.values).to(device) #.cuda()
        items = torch.LongTensor(train_data.movieId.values).to(device) #.cuda()
        ratings = torch.FloatTensor(train_data.rating.values).to(device) #.cuda()

        preds = model(users, items)
        loss = loss_fn(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['train_loss'].append(loss.item())

        if test_data is not None:
            test_loss = valid_loss(model, loss_fn, test_data, device)
            history['valid_loss'].append(test_loss)

        if verbose:
            if epoch == 1 or epoch % 10 == 0:
                if test_data is not None:
                    #print("train loss %.3f valid loss %.3f" % (loss.item(), test_loss))
                    #print(f"epoch [{epoch}/{epochs}] - train loss {loss.item():.3f} valid loss {test_loss:.3f}")
                    print('epoch %3d /%3d - train loss: %5.2f, val loss: %5.2f' %(epoch, epochs, loss.item(), test_loss))
                else:
                    print(f"epoch [{epoch}/{epochs}] - train loss {loss.item():.3f}")

    if save:
        torch.save(model.state_dict(), 'best_model.pth')

    return history

def valid_loss(model, loss_fn, test_data, device):
    model.eval()

    '''
    CSR Matrix ==========================
    rows, cols = test_data.nonzero()
    users = torch.LongTensor(rows).to(device) #.cuda()
    items = torch.LongTensor(cols).to(device) #.cuda()
    ratings = torch.FloatTensor(np.array(test_data[rows, cols]).squeeze()).to(device) #.cuda()
    '''
    # RATING DataFrame ==================
    users = torch.LongTensor(test_data.userId.values).to(device) #.cuda()
    items = torch.LongTensor(test_data.movieId.values).to(device) #.cuda()
    ratings = torch.FloatTensor(test_data.rating.values).to(device) #.cuda()

    preds = model(users, items)
    loss = loss_fn(preds, ratings)
    return loss.item()