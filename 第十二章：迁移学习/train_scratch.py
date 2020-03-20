import torch
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
from pokemon import Pokemon

batch_size = 32
lr = 1e-3
epochs = 10

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Pokemon(r'pokemon',224, mode='train')
val_db = Pokemon(r'pokemon',224, mode='val')
test_db = Pokemon(r'pokemon',224, mode='test')
train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_db,batch_size=batch_size)
test_loader = DataLoader(test_db,batch_size=batch_size)

def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct/total

if __name__ == '__main__':
    model = torchvision.models.resnet18().to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            logits = model(x)
            loss = criterion(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epochs % 2 == 0:
            val_acc = evalute(model,val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(),'best.mdl')

    print('best acc：', best_acc, 'best epoch：', best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt')
    test_acc = evalute(model, test_loader)
    print('test_acc', test_acc)