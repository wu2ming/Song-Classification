from torch import nn

#for callbacks
class EarlyStopping:
    def __init__(self, patience=5, deviation=0.01, learning=0.05):
        self.patience = patience
        self.deviation = deviation
        self.learning = learning
        self.counter1 = 0
        self.counter2 = 0
        self.best_loss = 0  # Initialize with a large value
        self.stop = False

    def __call__(self, train_loss, valid_loss):
        if(self.best_loss==0): self.best_loss = valid_loss

        # Check for overfitting condition
        if train_loss < valid_loss - self.deviation:
            self.counter1 += 1
            if self.counter1 >= self.patience:
                self.counter1 = 0  # Reset the counter
                self.stop = True
                print("Stopping because of overfitting")
        else:
            self.counter1 = 0

        # Check for stable learning rate condition
        if valid_loss > self.best_loss + self.learning:
            self.counter2 += 1
            if self.counter2 >= self.patience:
                self.counter2 = 0  # Reset the counter
                self.stop = True
                print("Stopping because of stable learning rate")
        else:
            self.best_loss = valid_loss
            self.counter2 = 0  # Reset the counter

        return self.stop

