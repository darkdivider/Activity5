from tqdm import tqdm
import pickle
import wandb
import torch

def train_model(config):
    print(f'Training on {config.device}')
    if config.wandb_log:
        wandb.login()
        run=wandb.init(
            project = config.name,
            config = {
                'optimizer': config.optimizer,
                'model': config.model
            }
        )
    for _ in range(config.epochs):
        best_loss=float('inf')
        losses = 0
        for X, y in tqdm(config.train_dataloader):
            config.model.train()
            X=X.to(config.device)
            y=y.to(config.device)
            output = config.model(X)
            loss = config.criterion(output, y)
            config.optimizer.zero_grad()
            loss.backward()
            config.optimizer.step()
            acc = config.metric(output, y)
            losses += loss.item()
        losses = losses/config.train_dataloader.dataset.__len__()
        if best_loss<losses:
            config.model.load_state_dict(torch.load(config.model_file))
        else:
            best_loss=losses
            torch.save(config.model.state_dict(), config.model_file)
        acc = config.metric.compute()
        print(f'Train_Loss: {losses:0.3g} | Train_Accuracy: {acc:0.3g}')
        if config.wandb_log:
            wandb.log({'train_accuracy': acc, 'train_loss':losses})
        config.metric.reset()
        
        for X, y in tqdm(config.test_dataloader):
            config.model.eval()
            X=X.to(config.device)
            y=y.to(config.device)
            output = config.model(X)
            loss = config.criterion(output, y)
            acc = config.metric(output, y)
            losses += loss.item()
        losses = losses/config.test_dataloader.dataset.__len__()
        # import pdb;pdb.set_trace()
        acc = config.metric.compute()
        print(f'Test_Loss: {losses:0.3g} | Test_Accuracy: {acc:0.3g}')
        if config.wandb_log:
            wandb.log({'test_accuracy': acc, 'test_loss':losses})
        config.metric.reset()
    config.scheduler.step()
    with open(config.model_file, 'wb') as file:
        pickle.dump(config.model,file)