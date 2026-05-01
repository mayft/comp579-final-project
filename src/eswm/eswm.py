import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_generation import Environment
import torch.nn.functional as f


class OpenWorld(nn.Module):
    def __init__(self):
        super(OpenWorld,self).__init__()
        self.source_embed = nn.Embedding(4, 128,padding_idx=0)
        self.end_embed = nn.Embedding(4, 128,padding_idx=0)
        self.action_embed = nn.Embedding(8, 768,padding_idx=0)

    def forward(self,x,mask=None):
        x=x+2
        #mask = (mask==0)
        start = torch.concat([self.source_embed(x[:, :, i]) for i in range(0, 6)], dim=-1)
        #start[:, -1] = start[:, -1] * mask[:, 0:1]

        action = self.action_embed(x[:, :, 6])
        #action[:, -1] = action[:, -1] * mask[:, 1:2]

        end = torch.concat([self.end_embed(x[:, :, i]) for i in range(7, 13)], dim=-1)
        #end[:, -1] = end[:, -1] * mask[:, 2:]

        return torch.mean(torch.stack([start, action, end], dim=0), dim=0)

class RandomWall(nn.Module):
    def __init__(self):
        super(RandomWall,self).__init__()
        self.source_embed = nn.Embedding(39,1024,padding_idx=0)
        self.action_embed = nn.Embedding(8,1024,padding_idx=0)
        self.end_embed = nn.Embedding(39,1024,padding_idx=0)

    def forward(self,x,mask=None):
        #mask = (mask == 0)
        x=x+2
        start = self.source_embed(x[:, :, 0])
        #start[:, -1] = start[:, -1] * mask[:, 0:1]
        action = self.action_embed(x[:, :, 1])
        #action[:, -1] = action[:, -1] * mask[:, 1:2]
        end = self.end_embed(x[:, :, 2])
        #end[:, -1] = end[:, -1] * mask[:, 2:]
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

    def forward(self, x,padding_mask=None,transition_mask=None):
        src = self.embedding(x,transition_mask)
        src = self.model(src,src_key_padding_mask=padding_mask)
        query = src[:,-1]
        out1 = (self.source(query))
        out2 = (self.action(query))
        out3 = (self.end(query))
        return out1,out2,out3
    
def get_batch(environment, batch_size,state_bins=True,test=False,device='cpu',kind='unseen'):
    x = environment.sample_environments(batch_size,test=test,kind=kind)
    x.append(torch.ones((environment.num_states,3)).numpy())
    x = nn.utils.rnn.pad_sequence([torch.tensor(i,device=device) for i in x],batch_first=True,padding_value=-2,padding_side='left')[:-1]

    #padding_mask = torch.tensor(padding_mask,device=device)
    padding_mask = (x == -2)
    #x=x.masked_fill(padding_mask,0)
    padding_mask = padding_mask[:,:,0]

    # mask queries
    q_mask = torch.randint(3, size=(x.shape[0],), device=device)
    q_mask = (f.one_hot(q_mask, 3)).ne(0)
    #q_mask = q_mask

    #convert to binary states
    source,action,end = x[:,:,0:1],x[:,:,1:2],x[:,:,2:]
    if state_bins: 
        source = source.bitwise_and(2**torch.arange(6,device=device)).ne(0).float()
        end = end.bitwise_and(2**torch.arange(6,device=device)).ne(0).float()

        mask = torch.concat([q_mask[:, 0:1].repeat([1, 6]), q_mask[:, 1:2], q_mask[:, 2:].repeat([1, 6])], dim=-1)
        x = torch.concat([source,action,end],dim=-1).long()
        y = [source[:, -1], action[:,-1,0],end[:,-1]] 
        padding_mask=None
    else:
        x=x.clone().long()
        mask=q_mask
        y = [source[:, -1,0], action[:,-1,0],end[:,-1,0]] 

    x[:, -1] = x[:, -1].masked_fill(mask, -1)

    return x,y,padding_mask,q_mask

def get_variable_batches(environment, batch_size,state_bins=True,test=False,device='cpu'):
    query_split= [0.68, 0.15, 0.17]
    x1, y1, pad1, mask1 = get_batch(environment, int(query_split[0]*batch_size),state_bins,test,device,'unseen')
    x2, y2, pad2, mask2 = get_batch(environment, int(query_split[1] * batch_size), state_bins, test, device, 'seen')
    x3, y3, pad3, mask3 = get_batch(environment, int(query_split[2] * batch_size), state_bins, test, device, 'unsolvable')
    y3[1]=y3[1].masked_fill(mask3[:,1:2].squeeze(1),6)
    y3[0]=y3[0].masked_fill(mask3[:,0:1].squeeze(1),37)
    y3[2]=y3[2].masked_fill(mask3[:,2:].squeeze(1),37)
    x = torch.concatenate([x1,x2,x3])
    pad = torch.concatenate([pad1,pad2,pad3])
    mask = torch.concatenate([mask1,mask2,mask3])
    y = [torch.concatenate([y1[0],y2[0],y3[0]]),
        torch.concatenate([y1[1], y2[1], y3[1]]),
        torch.concatenate([y1[2], y2[2], y3[2]])]
    return x,y,pad,mask

def accuracy(output, target):
    accuracies = []

    state_acc = ((output[0] > 0.5) == target[0]).float().mean(dim=-1)
    accuracies.append((state_acc==1).float().mean())

    accuracies.append((torch.argmax(output[1],dim=1) == target[1][:,0]).float().mean())

    state_acc = ((output[2] > 0.5) == target[2]).float().mean(dim=-1)
    accuracies.append((state_acc==1).float().mean())

    return accuracies

def masked_prediction(mask, output, target,state_bins,total=False):
    if state_bins:
        accuracy = torch.stack([((output[0] > 0.5) == target[0]).sum(dim=1)==6,
                                torch.argmax(output[1],dim=1)==target[1],
                                ((output[2] > 0.5) == target[2]).sum(dim=1)==6]).T       
    else:
        target=torch.stack(target,dim=1)
        predicted=torch.stack([torch.argmax(out,dim=1) for out in output]).T
        accuracy = target==predicted
    if total: return accuracy.float().mean(dim=0)
    return (mask*accuracy).sum(dim=0)/mask.sum(dim=0).clamp(min=1)

def train(environment,model_type,batch_size=128,device='cpu',epochs=10,num_layers=10,filename='eswm-results',pretrained=False,verbosity=100):
    if model_type=='open_arena':
        model_params={'embedder':OpenWorld()}
        loss_s= nn.BCEWithLogitsLoss()
        loss_a= nn.CrossEntropyLoss()
        state_bins=True
        batch = get_batch
    else:
        assert model_type == 'random_wall'
        model_params = {'embedder':RandomWall(),
                        'input_dim':1024,
                        'state_dim':38,
                        'num_actions':7
                        }
        loss_s= nn.CrossEntropyLoss()
        loss_a= nn.CrossEntropyLoss()
        state_bins=False
        batch = get_variable_batches
    
    model = ESWM(num_layers=num_layers,**model_params)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=460000)

    if pretrained:
        checkpoint = torch.load(f'{filename}.pth',weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start = checkpoint['epoch']
        file = open(f'{filename}.csv',mode='a')
        file.write('---training resumed---\n')
    else:
        file = open(f'{filename}.csv', mode='w')
        file.write('epoch,train-loss,train-loss-s,train-loss-a,train-loss-e,test-loss,test-loss-s,test-loss-a,test-loss-e,train-acc-s,train-acc-a,train-acc-e,test-acc-s,test-acc-a,test-acc-e\n')
        start = 0

    out_format = ','.join(['%d']+['%.8f'] * 14) + '\n'

    #openworld = OpenWorld()
    #model = ESWM(num_layers=num_layers,*model_params)
    #model.to(device)
    #loss_s= nn.BCEWithLogitsLoss()
    #loss_a= nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=460000)

    #test set
    x_t, y_t,pad_t,mask_t = batch(environment,2000,test=True,device=device,state_bins=state_bins)

    for epoch in tqdm(range(start,epochs)):
        model.train()
        src, target, padding_mask, transition_mask = batch(environment,batch_size,device=device,state_bins=state_bins)

        optimizer.zero_grad()
        output = model(src,padding_mask)
        loss1= torch.stack([loss_s(output[0][:,i],target[0][:,i]) for i in range(6)]).sum()
        loss2 = torch.stack([loss_s(output[2][:,i],target[2][:,i]) for i in range(6)]).sum()
        losses = [loss1,loss_a(output[1],target[1]),loss2]
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if epoch%verbosity==0:
                train_acc = masked_prediction(transition_mask,output,target,state_bins=state_bins)
                #mask =[transition_mask[:,0:1].repeat(1,6),transition_mask[:,1:2].repeat(1,6),transition_mask[:,2:].repeat(1,6)]
                #q_pred = [torch.masked_select(output[i], mask[i]).view(-1,6) for i in range(3)]
                #q_tar = [torch.masked_select(target[i], mask[i][:,:target[i].shape[1]]).view(-1,target[i].shape[1]) for i in range(3)]
                #train_acc = accuracy(q_pred,q_tar)

                model.eval()
                out = model(x_t, pad_t)
                test_acc = masked_prediction(mask_t,out,y_t,state_bins=state_bins)
                #m = [mask_t[:,0:1].repeat(1,6),mask_t[:,1:2].repeat(1,6),mask_t[:,2:].repeat(1,6)]
                #p = [torch.masked_select(out[i], m[i].ne(0)).view(-1,6) for i in range(3)]
                #t = [torch.masked_select(y_t[i], m[i][:,:y_t[i].shape[1]]).view(-1,y_t[i].shape[1]) for i in range(3)]
                loss1= torch.stack([loss_s(out[0][:,i],y_t[0][:,i]) for i in range(6)]).sum()
                loss2 = torch.stack([loss_s(out[2][:,i],y_t[2][:,i]) for i in range(6)]).sum()
                losses_t = [loss1,loss_a(out[1],y_t[1]),loss2]
                loss_t = torch.stack(losses_t).sum()
                #test_acc = accuracy(p,t)
                outputs = (epoch,loss,*losses,loss_t,*losses_t,*train_acc,*test_acc)
                file.write(out_format % outputs)
                file.flush()

            if epoch % 1000==0 or epoch+1==epochs:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(checkpoint, f'{filename}.pth')
                file.flush()

    file.close()
    return model

if __name__ == "__main__":
    open_env = Environment()
    #wall_env = Environment(side_length=4,add_wall=True,hidden=5,possible_states=37,query_all=True)
    #mod = train(env,epochs=200,filename='tester0',num_layers=6,device='mps')
    #env = Environment(side_length=4,add_wall=True,hidden=5,possible_states=37,query_all=True)
    train(open_env,epochs=4000,filename='src/tests/open3',num_layers=10,device='mps',model_type='open_arena',pretrained=False)
    #train(wall_env,epochs=2000,filename='src/tests/wall',num_layers=2,device='mps',model_type='random_wall')