import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datageneration import Environment
import torch.nn.functional as f

class OpenWorld(nn.Module):
    def __init__(self):
        super(OpenWorld,self).__init__()
        self.source_embed = nn.Embedding(2, 128)
        self.end_embed = nn.Embedding(2, 128)
        self.action_embed = nn.Embedding(6, 768)

    def forward(self,x,mask):
        start = torch.concat([self.source_embed(x[:, :, i]) for i in range(0, 6)], dim=-1)
        #print(start[0, -1, :])
        start[:, -1, :] = start[:, -1, :] * mask[:, :, 0]
        action = self.action_embed(x[:, :, 6])
        action[:, -1, :] = action[:, -1, :] * mask[:, :, 1]
        end = torch.concat([self.end_embed(x[:, :, i]) for i in range(7, 13)], dim=-1)
        end[:, -1, :] = end[:, -1, :] * mask[:, :, 2]
        #print(start[0,-1,:])
        return torch.mean(torch.stack([start, action, end], dim=0), dim=0)

class RandomWall(nn.Module):
    def __init__(self):
        super(RandomWall,self).__init__()
        self.source_embed = nn.Embedding(2,1024)
        self.action_embed = nn.Embedding(6,1024)
        self.end_embed = nn.Embedding(2,1024)

    def forward(self,x,mask):
        start = self.start_embed(x[:, :, 0])
        start[:, -1, :] = start[:, -1, :] * mask[:, :, 0]
        action = self.action_embed(x[:, :, 1])
        action[:, -1, :] = action[:, -1, :] * mask[:, :, 1]
        end = self.end_embed(x[:, :, 2])
        end[:, -1, :] = end[:, -1, :] * mask[:, :, 3]
        return torch.mean(torch.stack([start, action, end], dim=0), dim=0)

class ESWM(nn.Module):
    def __init__(self,embedder, num_layers=10,input_dim=768,nheads=8,feedforward=2048,dropout=0.1):
        super(ESWM,self).__init__()
        self.embedding = embedder
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim,nhead=nheads,dim_feedforward=feedforward,dropout=dropout,batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer=self.transformer,num_layers=num_layers)
        self.source = nn.Linear(input_dim,6)
        self.action = nn.Linear(input_dim, 6)
        self.end = nn.Linear(input_dim, 6)

    def forward(self, x,mask):
        src = self.embedding(x,mask)
        src = self.model(src)
        query = src[:,-1,:]
        out1 = (self.source(query))
        out2 = (self.action(query))
        out3 = (self.end(query))
        return torch.stack([out1,out2,out3])

def get_batch(env, batch_size,odd=False):
    x, m = env.generate_memory_bank(batch_size,odd)
    x = torch.tensor(x)
    y = x[:,-1,:]
    xmask  = f.one_hot(torch.tensor(m),3).unsqueeze(1)#.repeat(1,768).view(128,768,-1)

    act = f.one_hot(y[:,6],6)
    y = torch.stack([y[:,:6],act,y[:,7:]]).float()

    mask = torch.ones_like(y)
    for i in range(mask.shape[1]):
        #mask[i,6 * m[i]:6 * (m[i] + 1)] = 0
        mask[m[i],i,:] = 0
    mask = mask ==1
    return x,y, xmask,mask

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
    x_t, y_t,mx,m = get_batch(env,batch_size,odd=True)
    x_t = x_t.to(device_)
    y_t = y_t.to(device_)
    mx = mx.to(device_)

    model.train()
    for epoch in tqdm(range(epochs)):
        src, target, xmask,mask = get_batch(environment,batch_size)
        src=src.to(device_)
        target =target.to(device_)
        xmask = xmask.to(device_)
        optimizer.zero_grad()
        output = model(src,xmask)
        #print(xmask.shape)
        #print(mask.shape)
        p = [torch.masked_select(output[i], ~mask[i]).view(-1, 6) for i in range(3)]
        t = [torch.masked_select(target[i], ~mask[i]).view(-1, 6) for i in range(3)]

        loss = torch.stack([loss_s(p[0],t[0]),loss_a(p[1],t[1]),loss_s(p[2],t[2])])
        loss = loss.sum()
        #output = output.view(-1,output.shape[-1])
        #loss = criterion(p,t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch %10==0:
            with torch.no_grad():
                out = model(x_t, mx)
                p = [torch.masked_select(out[i], ~m[i]).view(-1, 6) for i in range(3)]
                t = [torch.masked_select(y_t[i], ~m[i]).view(-1, 6) for i in range(3)]
                acc = accuracy(p,t)
                outputs = (loss, *acc)
                file.write(out_format % outputs)
            #file.write(f'{loss},\n')
        if epoch % 1000==0:
            file.flush()
    torch.save(model.state_dict(), f'{filename}.pth')
    file.close()
    return model

if __name__ == "__main__":
    env = Environment(side_length=3,bin_states=True)
    mod = train(env,epochs=8000,filename='eswm-t10-v0')