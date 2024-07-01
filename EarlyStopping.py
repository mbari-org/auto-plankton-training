import torch
import os

class EarlyStopping:
    def __init__(self, patience = 7, verbose = False, delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        os.makedirs('temp', exist_ok = True)


    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, True)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}  ({-val_loss:.6f} is worse than {self.best_score:.6f})\n')
            if self.counter >= self.patience:
                return True 
        else:
            self.save_checkpoint(val_loss, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model, first_run = False):
        '''Saves model when validation loss decrease.'''
        if self.verbose and not first_run:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {-val_loss:.6f}). Saving model ...')
        elif self.verbose and first_run:
            print(f'Saving first run as baseline. (inf --> {self.best_score:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'temp/checkpoint.pt')

    def load_checkpoint(self, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('temp/checkpoint.pt'))
        model.to(device)
        return model
