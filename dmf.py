import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import normal_
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import wandb



########################################
# MLPLayers Class Definition
########################################
class MLPLayers(nn.Module):
    r"""
    MLPLayers constructs a multi-layer perceptron given a list of layer sizes.

    Args:
        layers (list): list containing the sizes of each layer. For example,
                       [input_size, hidden1, hidden2, ..., output_size].
        dropout (float): dropout probability applied before each linear layer.
        activation (str): activation function after each layer (default: 'leaky_relu').
        bn (bool): whether to use Batch Normalization after each linear layer.
        init_method (str): initialization method for weights (e.g., 'norm' for normal initialization).
        last_activation (bool): whether to apply the activation function on the last layer.
    """
    def __init__(self, layers, dropout=0.3, activation="leaky_relu", bn=False, init_method=None, last_activation=True):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation  
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        if self.init_method is not None:
            self.apply(self.init_weights)


    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            elif self.init_method == "xavier": 
                nn.init.xavier_normal_(module.weight) 
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)



# Helper function to return activation function
def activation_layer(activation, output_size):
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01) 
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    


########################################
# DMF Dataset
########################################
class DMFDataset(Dataset):
    def __init__(self, df, user_id_map, movie_id_map):
        self.users = torch.tensor(df['user_id'].map(user_id_map).values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_id'].map(movie_id_map).values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.users[idx],
            'movie_id': self.movies[idx],
            'rating': self.ratings[idx]
        }


########################################
# DMF Regressor with MLPLayers Integration (Regression Version)
########################################
class DMFRegressor(nn.Module):
    def __init__(self, 
                 num_users, num_movies,
                 global_interaction, 
                 user_embedding_size=32,
                 item_embedding_size=32,
                 user_hidden_sizes=[64, 32],
                 item_hidden_sizes=[64, 32],
                 dropout=0.3,
                 activation="leaky_relu",
                 bn=False,
                 init_method="norm"):
        """
        Args:
            num_users (int): Total number of users.
            num_movies (int): Total number of movies (items).
            global_interaction (torch.Tensor): Dense interaction matrix (shape: num_users x num_movies).
            user_embedding_size (int): Output size of the user linear layer.
            item_embedding_size (int): Output size of the item linear layer.
            user_hidden_sizes (list): List of hidden layer sizes for the user MLP.
            item_hidden_sizes (list): List of hidden layer sizes for the item MLP.
            dropout (float): Dropout probability.
            activation (str): Activation function to use (e.g., "relu").
            bn (bool): Whether to apply batch normalization.
            init_method (str): Weight initialization method.
        """

        super(DMFRegressor, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.register_buffer('global_interaction', global_interaction)
        self.user_linear = nn.Linear(num_movies, user_embedding_size, bias=False)
        self.item_linear = nn.Linear(num_users, item_embedding_size, bias=False)
        
        self.user_fc_layers = MLPLayers(
            [user_embedding_size] + user_hidden_sizes,
            dropout=dropout,
            activation=activation,
            bn=bn,
            init_method=init_method,
            last_activation=True
        )
        self.item_fc_layers = MLPLayers(
            [item_embedding_size] + item_hidden_sizes,
            dropout=dropout,
            activation=activation,
            bn=bn,
            init_method=init_method,
            last_activation=True
        )
        
        self.loss_fn = nn.HuberLoss(delta=0.5)
        
        
    def forward(self, user_indices, movie_indices):
        """
        Args:
            user_indices (torch.LongTensor): Batch of user indices (shape: [B]).
            movie_indices (torch.LongTensor): Batch of movie indices (shape: [B]).
        Returns:
            predictions (torch.Tensor): Predicted ratings (shape: [B]).
        """
        user_profile = self.global_interaction[user_indices]  
        u = self.user_linear(user_profile)  
        u = self.user_fc_layers(u)           
        
        item_profile = self.global_interaction.t()[movie_indices]  
        i = self.item_linear(item_profile)   
        i = self.item_fc_layers(i)          

        prediction = torch.mul(u, i).sum(dim=1)
        return prediction
    
    def calculate_loss(self, batch):
        """ Compute Huber Loss between predicted and actual ratings """
        user = batch['user_id']
        movie = batch['movie_id']
        rating = batch['rating']
        preds = self.forward(user, movie)
        loss = self.loss_fn(preds, rating)
        return loss
    
    def predict(self, batch):
        """
        Returns the predicted rating directly.
        """
        preds = self.forward(batch['user_id'], batch['movie_id'])
        return preds


########################################
# Training Function with Early Stopping & Wandb
########################################
        
def train_model_w_early_stopping(model, train_set, val_set, device, batch_size=64, num_epochs=10, lr=0.001, weight_decay=1e-4, patience=5, wandb=None, save_as = "best_model_state.pt"):
    """
    Train the model on the training set, using a validation set for early stopping.
    
    Args:
        model: The DMFModel instance.
        train_set: A PyTorch Dataset (e.g., DMFDataset) for training.
        val_set: A PyTorch Dataset for validation.
        device: The torch.device (e.g., "cuda" or "cpu").
        batch_size (int): Batch size for training.
        num_epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization) parameter.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    wandb= wandb
   
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            for key in batch:
                batch[key] = batch[key].to(device)
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch['rating'])
            
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_set)
        print(f"Epoch {epoch+1}/{num_epochs}  Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                loss = model.calculate_loss(batch)
                total_val_loss += loss.item() * len(batch['rating'])
        avg_val_loss = total_val_loss / len(val_set)
        print(f"Epoch {epoch+1}/{num_epochs}  Validation Loss: {avg_val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "patience_counter": patience_counter
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  
            patience_counter = 0
            print("  Validation loss improved. Saving model state.")
        else:
            patience_counter += 1
            print(f"  No improvement in validation loss for {patience_counter} epoch(s).")
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        model.train()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state with validation loss: {:.4f}".format(best_val_loss))

        torch.save(model.state_dict(), save_as)
        print("Saved best model state")
    
    wandb.finish() #To load the model model.load_state_dict(torch.load("best_model_state.pt"))
    

########################################
# Evaluation Function DMFRegressor
########################################

def evaluate_DMFRegressor(model, test_set, device, batch_size=64):
    """
    Evaluate the regression model on the test set.
    
    Args:
        model: The DMFModel (or any regression model) instance that predicts continuous ratings.
        test_set: A PyTorch Dataset (e.g., DMFDataset) for testing.
        device: The torch.device (e.g., "cuda" or "cpu").
        batch_size (int): Batch size for evaluation.
    
    Returns:
        average_loss: The average loss over the test set.
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        r2: R^2 score (coefficient of determination).
    """
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            for key in batch:
                batch[key] = batch[key].to(device)
            
            loss = model.calculate_loss(batch)
            total_loss += loss.item() * len(batch['rating'])
            
            preds = model.predict(batch)
            labels = batch['rating']  
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    average_loss = total_loss / len(test_set)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    std_error = np.std(all_preds - all_labels)
    print(f"Evaluation - Loss: {average_loss:.4f}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}, Std Error: {std_error:.4f}")
    
    return average_loss, mae, rmse, r2, std_error




