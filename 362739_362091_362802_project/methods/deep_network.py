import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import accuracy_fn, macrof1_fn
import time

## MS2!!

def tdecode(x):
    """
        One-hot decoder for torch tensors
    """
    return torch.argmax(x, dim=1)

class SimpleNetwork(nn.Module):
    """
    A network which does classification!
    """
    def __init__(self, input_size, num_classes, hidden_size=(120, 68, 36), conv_size=(6, 24)): 
        # best setup yet: convolution <|6, 24|> hidden <|120, 68, 36|> 
        # lr: 1e-4 epochs: 50 dropout: 0.6 conv_kernel:5 pool_kernel:2
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, conv_size[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_size[0], conv_size[1], kernel_size=5, padding=2)
        self.drop = nn.Dropout1d(p=0.6)
        self.fc1 = nn.Linear(conv_size[1] * (input_size // 4), hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], num_classes)

    def forward(self, x):
        """
        Takes as input the data x and outputs the 
        classification outputs.
        Args: 
            x (torch.tensor): shape (N, D)
        Returns:
            output_class (torch.tensor): shape (N, C) (logits)
        """
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = F.max_pool1d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool1d(F.relu(self.conv2(x)), kernel_size=2)
        # Flatten after convolutions and pooling over last 2 dimensions (channels, signal)
        x = self.drop(x).reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x)) #gelu or relu
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        # potentially F.softmax for last layer
        output_class = self.fc4(x)
        return output_class

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
        self.message = ""
        self.classification_criterion = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # SGD - around 300 iterations, lr: 1e-3 <> ADAM - around 60 iterations

    def train_all(self, dataloader_train, dataloader_val):
        """
        Method to iterate over the epochs. In each epoch, it should call the functions
        "train_one_epoch" (using dataloader_train) and "eval" (using dataloader_val).
        """
        start = time.time()
        print(f'Started Neural Network training with epochs:{self.epochs}, lr:{self.lr}...')
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train, ep)
            self.eval(dataloader_val)

            if (ep+1) % 50 == 0:
                print("Reduce Learning rate", end='\r')
                for g in self.optimizer.param_groups:
                    g["lr"] = g["lr"]*1 # 1 for ADAM, 0.8 otherwise
        end = time.time() 
        print(f'\nNeural Network Training Runtime |> {end-start}s')


    def train_one_epoch(self, dataloader, ep=0):
        """
        Method to train for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode!
        i.e. self.model.train()
        """
        self.model.train()
        
        for it, batch in enumerate(dataloader):
            # Load batch, break it down into joint and label
            x, err, y = batch


            # Run forward pass of batch.
            logits = self.model.forward(x) 

            # Compute loss (using 'criterion').
            loss = self.classification_criterion(logits, y)
            
            # Run backward pass.
            # torch.autograd.backward(loss) |> alternative option
            loss.backward()
            
            # Update the weights using optimizer.
            self.optimizer.step()
            
            # Zero-out the accumulated gradients.
            self.optimizer.zero_grad()

        self.message = '--> Ep {}/{}: loss train: {:.2f}, accuracy train: {:.2f},'.format(
                ep + 1, self.epochs, loss, accuracy_fn(tdecode(logits).numpy(), y.numpy()))

    def eval(self, dataloader):
        """
            Method to evaluate model using the validation dataset OR the test dataset.
            Don't forget to set your model to eval mode!
            i.e. self.model.eval()

            Returns:
                Note: N is the amount of validation/test data. 
                We return one torch tensor which we will use to save our results (for the competition!)
                results_class (torch.tensor): classification results of shape (N,)
        """
        self.model.eval()
        with torch.no_grad():
            acc_run = 0
            macrof1_run = 0
            results_class = torch.tensor([])
            for it, batch in enumerate(dataloader):
                # Get batch of data.
                x, err,  y = batch
                curr_bs = x.shape[0]
                results_class = torch.cat((results_class, tdecode(self.model(x))), axis=0)
                acc_run += accuracy_fn(tdecode(self.model(x)).numpy(), y.numpy()) * curr_bs
                macrof1_run += macrof1_fn(tdecode(self.model(x)).numpy(), y.numpy()) * curr_bs
            acc = acc_run / len(dataloader.dataset)
            macrof1 = macrof1_run / len(dataloader.dataset)

            print(self.message + 'accuracy test: {:.2f}, macro F1 score: {:.2f}'.format(acc, macrof1), end='\r')
        
        return results_class