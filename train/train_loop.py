import torch
from utils.earlystopping import EarlyStopping


loss_list = []
val_list  = []
def train_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    earlystopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(1, n_epochs + 1):
        loss_sum     = 0.0
        valloss_sum  = 0.0
        model.train()
        for blur, truth in train_loader:
            blur     = blur.to(device = device)
            truth    = truth.to(device = device)
            out, rec = model(blur) #, 0
            loss     = loss_fn(out, truth)
            optimizer.zero_grad()
            loss.backward(retain_graph=True) ###
            optimizer.step()
            loss_sum += loss.item()
            
        loss_list.append(loss_sum / len(train_loader))
        
        model.eval()
        with torch.no_grad():    
            for blur, truth in val_loader:
                blur        = blur.to(device = device)
                truth       = truth.to(device = device)
                out, rec    = model(blur) #, 0
                val_loss    = loss_fn(out, truth)
                valloss_sum += val_loss                
            val_list.append(valloss_sum / len(val_loader))
        if epoch == 1 or epoch % 5 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {val_list[-1]}')
        earlystopping((loss_sum / len(train_loader)), model)
        if earlystopping.early_stop:
            break