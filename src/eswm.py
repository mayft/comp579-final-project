import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_generation import Environment
import torch.nn.functional as f

class OpenWorld(nn.Module):
    def __init__(self):
        super(OpenWorld,self).__init__()
        self.source_embed = nn.Embedding(3, 128)
        self.end_embed = nn.Embedding(3, 128)
        self.action_embed = nn.Embedding(7, 768)

    def forward(self,x,mask):
        mask = (mask==0)
        start = torch.concat([self.source_embed(x[:, :, i]) for i in range(0, 6)], dim=-1)
        start[:, -1] = start[:, -1] * mask[:, 0:1]

        action = self.action_embed(x[:, :, 6])
        action[:, -1] = action[:, -1] * mask[:, 1:2]

        end = torch.concat([self.end_embed(x[:, :, i]) for i in range(7, 13)], dim=-1)
        end[:, -1] = end[:, -1] * mask[:, 2:]

        return torch.mean(torch.stack([start, action, end], dim=0), dim=0)

class RandomWall(nn.Module):
    def __init__(self):
        super(RandomWall,self).__init__()
        self.source_embed = nn.Embedding(37,1024)
        self.action_embed = nn.Embedding(6,1024)
        self.end_embed = nn.Embedding(37,1024)

    def forward(self,x,mask):
        start = self.source_embed(x[:, :, 0])
        start[:, -1, :] = start[:, -1, :] * mask[:, 0:1]
        action = self.action_embed(x[:, :, 1])
        action[:, -1, :] = action[:, -1, :] * mask[:, 1:2]
        end = self.end_embed(x[:, :, 2])
        end[:, -1, :] = end[:, -1, :] * mask[:, 2:]
        return torch.mean(torch.stack([start, action, end], dim=0), dim=0)

class ESWM(nn.Module):
    def __init__(self,embedder, num_layers=10,input_dim=768,num_heads=8,feedforward=2048,dropout=0.1,state_dim=6,num_actions=6):
        super(ESWM,self).__init__()
        self.embedding = embedder
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim,nhead=num_heads,dim_feedforward=feedforward,dropout=dropout,batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer=self.transformer,num_layers=num_layers)
        self.source = nn.Linear(input_dim,state_dim)
        self.action = nn.Linear(input_dim, num_actions)
        self.end = nn.Linear(input_dim, state_dim)

    def forward(self, x,padding_mask,transition_mask):
        src = self.embedding(x,transition_mask)
        src = self.model(src,src_key_padding_mask=padding_mask)
        query = src[:,-1]
        out1 = (self.source(query))
        out2 = (self.action(query))
        out3 = (self.end(query))
        return [out1,out2,out3]

def get_batch(environment, batch_size,state_bins=True,test=False,device='cpu'):
    x, padding_mask = environment.generate_memory_bank(batch_size,test=test)
    x = torch.tensor(x,device=device)
    padding_mask = torch.tensor(padding_mask,device=device)

    source,action,end = x[:,:,0:1],x[:,:,1:2],x[:,:,2:]

    #convert to binary states
    if state_bins:
        source = source.bitwise_and(2**torch.arange(6,device=device)).ne(0).float()
        end = end.bitwise_and(2**torch.arange(6,device=device)).ne(0).float()
        x = torch.concat([source,action,end],dim=-1)

    x = x.long()
    y= [source[:,-1],action[:,-1],end[:,-1]]

    #mask queries
    q_mask = torch.randint(3, size=(x.shape[0],),device=device)
    q_mask = (f.one_hot(q_mask, 3)).ne(0)
    data_mask = torch.concat([q_mask[:,0:1].repeat([1,6]), q_mask[:,1:2], q_mask[:,2:].repeat([1,6])],dim=-1)
    x[:, -1] = x[:, -1].masked_fill(data_mask, 0)

    return x,y,padding_mask,q_mask

def accuracy(output, target):
    accuracies = []

    state_acc = ((output[0] > 0.5) == target[0]).float().mean(dim=-1)
    accuracies.append((state_acc==1).float().mean())

    accuracies.append((torch.argmax(output[1],dim=1) == target[1][:,0]).float().mean())

    state_acc = ((output[2] > 0.5) == target[2]).float().mean(dim=-1)
    accuracies.append((state_acc==1).float().mean())

    return accuracies

def train(environment,batch_size=128,device='cpu',epochs=10,num_layers=10,filename='eswm-results'):
    file = open(f'{filename}.csv', mode='w')
    file.write('epoch,train-loss,train-loss-s,train-loss-a,train-loss-e,test-loss,test-loss-s,test-loss-a,test-loss-e,train-acc-s,train-acc-a,train-acc-e,test-acc-s,test-acc-a,test-acc-e\n')
    out_format = ','.join(['%d']+['%.8f'] * 14) + '\n'

    openworld = OpenWorld()
    model = ESWM(num_layers=num_layers,embedder=openworld)
    model.to(device)
    loss_s= nn.BCEWithLogitsLoss()
    loss_a= nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=460000)

    #test set
    x_t, y_t,pad_t,mask_t = get_batch(environment,1000,test=True,device=device)

    for epoch in tqdm(range(epochs)):
        model.train()
        src, target, padding_mask, transition_mask = get_batch(environment,batch_size,device=device)

        optimizer.zero_grad()
        output = model(src,padding_mask,transition_mask)
        losses = [loss_s(output[0],target[0]),loss_a(output[1],target[1][:,0]),loss_s(output[2],target[2])]
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            mask =[transition_mask[:,0:1].repeat(1,6),transition_mask[:,1:2].repeat(1,6),transition_mask[:,2:].repeat(1,6)]
            q_pred = [torch.masked_select(output[i], mask[i]).view(-1,6) for i in range(3)]
            q_tar = [torch.masked_select(target[i], mask[i][:,:target[i].shape[1]]).view(-1,target[i].shape[1]) for i in range(3)]
            train_acc = accuracy(q_pred,q_tar)

            if epoch %10==0:
                model.eval()
                out = model(x_t, pad_t,mask_t)
                m = [mask_t[:,0:1].repeat(1,6),mask_t[:,1:2].repeat(1,6),mask_t[:,2:].repeat(1,6)]
                p = [torch.masked_select(out[i], m[i].ne(0)).view(-1,6) for i in range(3)]
                t = [torch.masked_select(y_t[i], m[i][:,:y_t[i].shape[1]]).view(-1,y_t[i].shape[1]) for i in range(3)]
                losses_t = [loss_s(out[0],y_t[0]),loss_a(out[1],y_t[1][:,0]),loss_s(out[2],y_t[2])]
                loss_t = torch.stack(losses_t).sum()
                test_acc = accuracy(p,t)
                outputs = (epoch,loss,*losses,loss_t,*losses_t,*train_acc,*test_acc)
                file.write(out_format % outputs)
                file.flush()

            if epoch % 1000==0 or epoch+1==epochs:
                torch.save(model.state_dict(), f'{filename}.pth')
                file.flush()

    file.close()
    return model

if __name__ == "__main__":
    env = Environment()
    mod = train(env,epochs=200,filename='tester0',num_layers=6,device='mps')