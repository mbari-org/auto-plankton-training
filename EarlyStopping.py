# Author: Steven Patrick
import torch
import os

class EarlyStopping:
    """Class for implementing early stopping during model training to prevent overfitting.
    
    Early stopping halts training if the validation loss does not improve for a certain number
    of consecutive epochs, which can save time and computational resources.
    
    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        verbose (bool): If True, prints detailed messages about the training progress.
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
        counter (int): Counts the number of epochs with no improvement.
        best_score (float): The best recorded score (negative validation loss) during training.
        early_stop (bool): Indicator of whether early stopping has been triggered.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """Initializes EarlyStopping with user-defined patience, verbosity, and delta.
        
        Args:
            patience (int, optional): Number of epochs with no improvement after which training stops. Defaults to 7.
            verbose (bool, optional): If True, prints messages about training progress and when the model is saved. Defaults to False.
            delta (float, optional): Minimum change in validation loss to qualify as an improvement. Defaults to 0.
        """        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        os.makedirs('temp', exist_ok=True)

    def __call__(self, val_loss, model):
        """Tracks validation loss and decides whether to trigger early stopping.
        
        Early stopping is triggered only when validation loss does not improve for
        'patience' consecutive epochs.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.

        Returns:
            bool: True if early stopping is triggered, otherwise False.
        """    
        score = -val_loss  # Using negative loss to treat lower losses as better scores.

        if self.best_score is None:
            # Save the model at the first run.
            self.best_score = score
            self.save_checkpoint(val_loss, model, first_run=True)
        elif score < self.best_score + self.delta:
            # No improvement in validation loss.
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}  ({-val_loss:.6f} is worse than {self.best_score:.6f})\n')
            if self.counter >= self.patience:
                # Stop training if no improvement for 'patience' consecutive epochs.
                self.early_stop = True
                return True  # Trigger early stopping.
        else:
            # Improvement in validation loss.
            self.save_checkpoint(val_loss, model)
            self.best_score = score
            self.counter = 0  # Reset counter if there is an improvement

        return False  # Continue training if early stopping not triggered


    def save_checkpoint(self, val_loss, model, first_run=False):
        """Saves the current model checkpoint if the validation loss decreases.
        
        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
            first_run (bool, optional): True if it's the first time saving the model. Defaults to False.
        """        
        if self.verbose and not first_run:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {-val_loss:.6f}). Saving model ...')
        elif self.verbose and first_run:
            print(f'Saving first run as baseline. (inf --> {self.best_score:.6f}). Saving model ...')
        
        # Saving the model
        torch.save(model.state_dict(), 'temp/checkpoint.pt')

    def load_checkpoint(self, model):
        """Loads the model from the last saved checkpoint.
        
        Args:
            model (torch.nn.Module): The model to load the checkpoint into.

        Returns:
            torch.nn.Module: The model with the loaded checkpoint.
        """        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('temp/checkpoint.pt'))
        model.to(device)
        return model
