import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_generation import Environment
import torch.nn.functional as f

class OpenWorld(nn.Module):
    def __init__(self):
        super(OpenWorld,self).__init__()
        self.source_embed = nn.Embedding(2, 128)
        self.end_embed = nn.Embedding(2, 128)
        self.action_embed = nn.Embedding(6, 768)

    def forward(self,x,mask):
        start = torch.concat([self.source_embed(x[:, :, i]) for i in range(0, 6)], dim=-1)
        start[:, -1] = start[:, -1] * mask[:, 0:1]
        action = self.action_embed(x[:, :, 6])
        action[:, -1, :] = action[:, -1, :] * mask[:,  1:2]
        end = torch.concat([self.end_embed(x[:, :, i]) for i in range(7, 13)], dim=-1)
        end[:, -1, :] = end[:, -1, :] * mask[:,  2:]
        #print(start[0,-1,:])
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
    def __init__(self,embedder, num_layers=10,input_dim=768,nheads=8,feedforward=2048,dropout=0.1,state_dim=6):
        super(ESWM,self).__init__()
        self.embedding = embedder
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim,nhead=nheads,dim_feedforward=feedforward,dropout=dropout,batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer=self.transformer,num_layers=num_layers)
        self.source = nn.Linear(input_dim,state_dim)
        self.action = nn.Linear(input_dim, 6)
        self.end = nn.Linear(input_dim, state_dim)

    def forward(self, x,src_mask,transition_mask):
        src = self.embedding(x,transition_mask)
        src = self.model(src,src_key_padding_mask=src_mask)
        query = src[:,-1,:]
        out1 = (self.source(query))
        out2 = (self.action(query))
        out3 = (self.end(query))
        return [out1,out2,out3]

def get_batch(env, batch_size,state_bins=True):
    x, padding_mask = env.generate_memory_bank(batch_size)
    x = torch.tensor(x).long()
    padding_mask=torch.tensor(padding_mask).float()

    #y = x[:,-1,:]
    s1,s2 = x[:,:,0].unsqueeze(-1),x[:,:,2].unsqueeze(-1)
    act = f.one_hot(x[:,:, 1], 6)
    if state_bins:
        s1 = s1.bitwise_and(2**torch.arange(6)).ne(0).byte()
        s2 = s2.bitwise_and(2**torch.arange(6)).ne(0).byte()
        x = torch.concat([s1,x[:,:,1].unsqueeze(-1),s2],dim=-1)

    y = torch.concat([s1,act,s2],dim=-1)[:,-1,:].float()

    xmask = torch.randint(3,size=(x.shape[0],))
    xmask = f.one_hot(xmask,3)
    return x,y, xmask,padding_mask

def accuracy(output, target):
    accuracy = []

    med = ((output[0] > 0.5) == target[0]).float().mean(dim=-1)
    #accuracy.append(med.mean())
    accuracy.append((med==1).float().mean())
    accuracy.append((torch.argmax(output[1],dim=1) == torch.argmax(target[1],dim=1)).float().mean())
    med = ((output[2] > 0.5) == target[2]).float().mean(dim=-1)
    #accuracy.append(med.mean())
    accuracy.append((med==1).float().mean())
    accuracy.append((accuracy[0]*len(output[0])+accuracy[1]*len(output[1])+accuracy[2]*len(output[2]))/128)
    return accuracy

def train(environment,batch_size=128,device='mps',epochs=10,num_layers=10,filename='eswm-results'):
    file = open(f'{filename}.csv', mode='w')
    file.write('loss,source_acc,action_acc,end_acc,total_acc\n')
    out_format = ','.join(['%.8f'] * 5) + '\n'
    openworld = OpenWorld()
    model = ESWM(num_layers=num_layers,embedder=openworld)
    device_=torch.device(device)
    model.to(device_)
    #criterion = nn.CrossEntropyLoss()
    loss_s= nn.BCEWithLogitsLoss()
    loss_a= nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=460000)
    x_t, y_t,mx,mp = get_batch(env,batch_size)
    x_t = x_t.to(device_)
    y_t = y_t.to(device_)
    mx = mx.to(device_)
    mp = mp.to(device_)

    model.train()
    for epoch in tqdm(range(epochs)):
        src, target, transition_mask,src_mask = get_batch(environment,batch_size)
        src=src.to(device_)
        target =target.to(device_)
        transition_mask = transition_mask.to(device_)
        src_mask=src_mask.to(device_)
        optimizer.zero_grad()
        output = model(src,src_mask,transition_mask)
        #print(xmask.shape)
        #print(mask.shape)
        m =torch.stack([transition_mask] * 6).permute([2, 1, 0])
        p = [torch.masked_select(output[i], m[i].ne(0)).view(-1, 6) for i in range(3)]
        t = [torch.masked_select(target[:,i*6:(i+1)*6], m[i].ne(0)).view(-1, 6) for i in range(3)]

        loss = torch.stack([loss_s(p[0],t[0]),loss_a(p[1],t[1]),loss_s(p[2],t[2])])
        loss = loss.sum()

        #output = output.view(-1,output.shape[-1])
        #loss = criterion(p,t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch %10==0:
            with torch.no_grad():
                out = model(x_t, mp,mx)
                m=torch.stack([mx]*6).permute([2,1,0])
                p = [torch.masked_select(out[i], m[i].ne(0)).view(-1, 6) for i in range(3)]
                t = [torch.masked_select(y_t[:,i*6:(i+1)*6], m[i].ne(0)).view(-1, 6) for i in range(3)]
                acc = accuracy(p,t)
                outputs = (loss, *acc)
                file.write(out_format % outputs)

        if epoch % 1000==0:
            file.flush()
            torch.save(model.state_dict(), f'{filename}.pth')
    file.close()
    return model

if __name__ == "__main__":
    env = Environment()
    mod = train(env,epochs=1000,filename='tester2')