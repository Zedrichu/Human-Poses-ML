import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import accuracy_fn, macrof1_fn

## MS2!!


class SimpleNetwork(nn.Module):
    """
    A network which does both classification and regression!
    """
    def __init__(self, input_size, num_classes, regression_output_size, hidden_size=32):
        super(SimpleNetwork, self).__init__()

        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

    def forward(self, x):
        """
        Takes as input the data x and outputs both the 
        classification and regression outputs.
        Args: 
            x (torch.tensor): shape (N, D)
        Returns:
            output_class (torch.tensor): shape (N, C) (logits)
            output_reg: shape (N, regression_target_size) 
        """

        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return output_class, output_reg

class Trainer(object):

    """
        Trainer class for the deep network.
    """

    def __init__(self, model, lr, epochs, beta=100):
        """
        """
        self.lr = lr
        self.epochs = epochs
        self.model= model
        self.beta = beta

        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader_train, dataloader_val):
        """
        Method to iterate over the epochs. In each epoch, it should call the functions
        "train_one_epoch" (using dataloader_train) and "eval" (using dataloader_val).
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train)
            self.eval(dataloader_val)

            if (ep+1) % 50 == 0:
                print("Reduce Learning rate")
                for g in self.optimizer.param_groups:
                    g["lr"] = g["lr"]*0.8


    def train_one_epoch(self, dataloader):
        """
        Method to train for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        This method should compute TWO losses (unlike what we have seen so far)
        One loss should be for classification, one loss should be for training. 
        You can sum these two losses for one overall loss. 
        Ex:
            loss1 = self.classification_criterion(output_class, labels)
            loss2 = self.regression_criterion(output_reg, regression_targets)
            loss = loss1 + loss2
        Don't forget to set your model to training mode!
        i.e. self.model.train()
        """
        
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

    def eval(self, dataloader):
        """
            Method to evaluate model using the validation dataset OR the test dataset.
            Don't forget to set your model to eval mode!
            i.e. self.model.eval()

            Returns:
                Note: N is the amount of validation/test data. 
                We return two torch tensors which we will use to save our results (for the competition!)
                results_class (torch.tensor): classification results of shape (N,)
                results_reg  (torch.tensor): regression results of shape (N, regression_target_size)

        """
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
        
        return results_class, results_reg