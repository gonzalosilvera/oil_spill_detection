# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy

# Define the convolutional layer with batch normalization and ReLU activation
def conv(in_channels, out_channels, kernel_size=3, padding=1):
    """
    Create a convolutional layer followed by batch normalization and ReLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        padding (int): Padding added to all four sides of the input. Default is 1.
    
    Returns:
        nn.Sequential: A sequential container with Conv2d, BatchNorm2d, and ReLU layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

# Define the CNN class for image segmentation
class CNN(LightningModule):
    
    def __init__(self, in_channels=1, out_channels=1):
        """
        Initialize the CNN model with encoder and decoder layers.
        
        Args:
            in_channels (int): Number of input channels. Default is 1.
            out_channels (int): Number of output channels. Default is 1.
        """
        super(CNN, self).__init__()
        
        # Define the number of channels for each layer
        c = [16, 32, 64, 128]
        
        # Define the encoder layer (downsampling)
        self.encoder = nn.Sequential(
            conv(in_channels, c[0]),  # First convolutional block
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by a factor of 2
            conv(c[0], c[1]),  # Second convolutional block
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by a factor of 2
            conv(c[1], c[2]),  # Third convolutional block
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by a factor of 2
        )
        
        # Define the decoder layer (upsampling)
        self.decoder = nn.Sequential(
            conv(c[2], c[1]),  # First upsampling block
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # Upsample by a factor of 2
            conv(c[1], c[0]),  # Second upsampling block
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # Upsample by a factor of 2
            conv(c[0], out_channels),  # Final convolutional block
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # Upsample to original size
        )
     
    # Define the forward method
    def forward(self, x):
        """
        Perform a forward pass through the CNN model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying the encoder and decoder.
        """
        x = self.encoder(x)  # Pass input through the encoder
        x = self.decoder(x)  # Pass encoded features through the decoder
        x = nn.Sigmoid()(x)  # Apply sigmoid activation to the output
        return x

# Define the OilSpillSegmentation class for training and testing the model
class OilSpillSegmentation(LightningModule):
    def __init__(self, learning_rate=0.001, seed=None):
        """
        Initialize the OilSpillSegmentation module with a UNet model, loss function, 
        learning rate, and accuracy metric.
        
        Args:
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        """
        super().__init__()
        self.model = CNN()  # UNet model for image segmentation
        self.save_hyperparameters() # Save hyperparameters for later use
        self.loss_function = nn.MSELoss()  # Mean Squared Error loss function
        self.learning_rate = learning_rate  # Learning rate for the optimizer
        self.accuracy_metric = BinaryAccuracy()  # Metric to calculate binary classification accuracy
        
        # Initialize variables to track true positives, false positives, and false negatives for test metrics
        self.test_tp = 0
        self.test_fp = 0
        self.test_fn = 0
      
        # Initialize threshold for converting probabilities to binary predictions
        self.threshold = 0.5

        # buffers to store validation outputs and targets for threshold optimization
        self.val_outputs = []
        self.val_targets = []

    # Define the forward method
    def forward(self, x):
        """
        Perform a forward pass through the UNet model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)

    # Define the training_step method
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """
        data, target = batch
        output = self.forward(data)
        loss = self.loss_function(output, target)

        # Convert predictions to binary using current threshold
        preds = (output > self.threshold).int()
        target = target.int()

        # Calculate accuracy using the BinaryAccuracy metric
        acc = self.accuracy_metric(preds, target)

        # Log training loss and accuracy for both step and epoch, and display in progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    # Define the validation_step method
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        """
        data, target = batch
        output = self.forward(data)
        loss = self.loss_function(output, target)

        # Save outputs and targets for threshold optimization
        self.val_outputs.append(output.detach().cpu())
        self.val_targets.append(target.detach().cpu())

        # Metrics with current threshold (just for logging)
        preds = (output > self.threshold).int()
        target = target.int()

        # Calculate accuracy using the BinaryAccuracy metric
        acc = self.accuracy_metric(preds, target)

        # Log validation loss and accuracy for the epoch, and display in progress bar
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}

    # Auto-threshold at the end of each validation epoch
    def on_validation_epoch_end(self):
        """
        Compute best threshold based on validation IoU
        """
        if len(self.val_outputs) == 0:
            return

        outputs = torch.cat(self.val_outputs)
        targets = torch.cat(self.val_targets).int()

        thresholds = torch.linspace(0.1, 0.9, 17)

        best_iou = 0
        best_t = self.threshold

        # Iterate over potential thresholds to find the one that maximizes IoU
        for t in thresholds:
            preds = (outputs > t).int()

            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()

            denom = tp + fp + fn

            if denom == 0:
                iou = torch.tensor(1.0)
            else:
                iou = tp / denom

            if iou > best_iou:
                best_iou = iou
                best_t = t.item()

        # Update threshold
        self.threshold = best_t

        print(f"\n[Auto Threshold] Best threshold: {best_t:.3f} | IoU: {best_iou:.4f}")

        # Clear buffers
        self.val_outputs.clear()
        self.val_targets.clear()

    # Define the on_test_epoch_start method to reset test metrics at the beginning of each test epoch
    def on_test_epoch_start(self):
        self.test_tp = 0
        self.test_fp = 0
        self.test_fn = 0
        
    # Define the test_step method
    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.
        """
        data, target = batch
        output = self.forward(data)
        loss = self.loss_function(output, target)

        # Use the best threshold found in validation
        preds = (output > self.threshold).int()
        target = target.int()

        # Calculate accuracy using the BinaryAccuracy metric
        accuracy = self.accuracy_metric(preds, target)
        
        # Update true positives, false positives, and false negatives for test metrics
        tp = ((preds == 1) & (target == 1)).sum().item()
        fp = ((preds == 1) & (target == 0)).sum().item()
        fn = ((preds == 0) & (target == 1)).sum().item()

        # Update test metrics
        self.test_tp += tp
        self.test_fp += fp
        self.test_fn += fn

        # Log test loss and accuracy for the epoch, and display in progress bar
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", accuracy, on_epoch=True, prog_bar=True)

    # Define the on_test_epoch_end method to calculate and log the final IoU metric at the end of each test epoch
    def on_test_epoch_end(self):
        denom = self.test_tp + self.test_fp + self.test_fn

        if denom == 0:
            iou = 1.0
        else:
            iou = self.test_tp / denom

        self.log("test_iou", iou)
        
    # Define the configure_optimizers method
    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)