import torch
from torch import nn    
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm


class Autoencoder(nn.Module):
    def __init__(self, n_features, hidden_layers_size, latent_size, activation, last_activation):
        super().__init__()
        self._n_features = n_features
        self._hidden_layers_size = hidden_layers_size
        self._latent_size = latent_size
        self._activation = activation
        self._last_activation = last_activation
        
        self._build_encoder()
        self._build_decoder()
    
    def _get_activation_layer(self, activation_name):
        if activation_name.lower() == "relu":
            return nn.ReLU()
        if activation_name.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            ValueError("Invalid activation layer")
    
    def _build_encoder(self):
        layer_list = []
        
        # add input layer
        layer_list += [nn.Linear(in_features=self._n_features, out_features=self._hidden_layers_size[0]),
                      self._get_activation_layer(self._activation)]
        
        # add hidden layers and activations
        for idx in range(len(self._hidden_layers_size) - 1):
            hidden_layer = nn.Linear(in_features=self._hidden_layers_size[idx], out_features=self._hidden_layers_size[idx+1])
            activation_layer = self._get_activation_layer(self._activation)
            
            layer_list += [hidden_layer, activation_layer]

        # add latent layer
        latent_layer = nn.Linear(in_features=self._hidden_layers_size[-1], out_features=self._latent_size)
        activation_layer = self._get_activation_layer(self._activation)
        layer_list += [latent_layer, activation_layer]
        
        self._encoder = nn.Sequential(*layer_list)
    
    def _build_decoder(self):
        layer_list = []
        
        layer_list += [nn.Linear(in_features=self._latent_size, out_features=self._hidden_layers_size[-1]),
                       self._get_activation_layer(self._activation)]
        
        n_layers = len(self._hidden_layers_size)
        
        for idx in range(len(self._hidden_layers_size) - 1):
            activation_layer = self._get_activation_layer(self._activation)   
            hidden_layer = nn.Linear(in_features=self._hidden_layers_size[n_layers - idx - 1],
                                     out_features=self._hidden_layers_size[n_layers - idx- 2])
            
            layer_list += [hidden_layer, activation_layer]
            
        # output layer
        layer_list += [nn.Linear(in_features=self._hidden_layers_size[0], out_features=self._n_features),
                      self._get_activation_layer(self._last_activation)]
            
        self._decoder = nn.Sequential(*layer_list)
    
    
    def encode(self, X):
        return self._encoder(X)
    
    def decode(self, X):
        return self._decoder(X)
    
    def forward(self, X):
        X = torch.FloatTensor(X)
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        
        return decoded
    
    def anomaly_score(self, X, anomaly_threshold=None, beta=100):
        """ Calculates reconstruction error of the sample X"""
        X = torch.Tensor(X)
        X_pred = self.forward(X)
        error = torch.sum((X - X_pred) ** 2, axis=1)

        if anomaly_threshold is not None:
            anomaly_score = nn.Softplus(beta=beta)(error - anomaly_threshold)
        else:
            anomaly_score = error

        return anomaly_score.detach().numpy()

    def anomaly_score_multi(self, X):
        """ Temporary added to be used in PSO to Calculates reconstruction error of multiple samples X"""
        X = torch.Tensor(X)
        X_pred = self.forward(X)
        error = torch.sum((X - X_pred) ** 2, axis=1)
        error = error.detach().numpy()
        return error

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class AutoencoderDataset(Dataset):
    """
    Class implementing a torch's Dataset - used for training the autoencoder
    """
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        return x

class AutoencoderLearner:
    """
    Static class containing methods for training autoencoder
    """
    
    @staticmethod
    def fit(model, data_loader, device, optimizer, loss_function):
        running_loss = .0
        model.train()

        for idx, inputs in tqdm(enumerate(data_loader), total=data_loader.__len__(), disable=True):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())      
            loss = loss_function(preds , inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss

        train_loss = running_loss / len(data_loader)
        train_loss = train_loss.detach().numpy()
        return train_loss
    
    @staticmethod
    def validate(model, data_loader, device, optimizer, loss_function):
        running_loss = .0
        model.eval()

        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                preds = model(inputs.float())
                loss = loss_function(preds, inputs)
                running_loss += loss

            valid_loss = running_loss / len(data_loader)
            valid_loss = valid_loss.detach().numpy()

            return valid_loss

    @staticmethod
    def run_training(model,
                     optimizer,
                     loss_function,
                     train_loader,
                     test_loader,
                     epochs=1,
                     device_name="cpu",
                     early_stopping=False,
                     early_stopping_patience=5,
                     early_stopping_delta=1e-5,
                     early_stopping_checkpoint='checkpoint.pt'
                    ):
        device = torch.device(device_name)

        train_losses = []
        valid_losses = []

        t = tqdm(range(epochs), desc='Training for %i epochs' % epochs, leave=True)

        # iniatlize loss to extremely high value (for early stopping)
        best_loss = 1e16
        epochs_without_progress = 0

        for epoch in t:
            # train
            train_loss = AutoencoderLearner.fit(model, train_loader, device, optimizer, loss_function)
            train_losses.append(train_loss)
            # validate
            valid_loss = AutoencoderLearner.validate(model, test_loader, device, optimizer, loss_function)
            valid_losses.append(valid_loss)

            if early_stopping:
                if (valid_loss < best_loss - early_stopping_delta):
                    best_loss = valid_loss
                    epochs_without_progress = 0
                    model.save_weights(early_stopping_checkpoint)
                else:
                    epochs_without_progress +=1

                if epochs_without_progress >= early_stopping_patience:
                    print("Not enough progress in last %i epochs, end of training." % early_stopping_patience)
                    model.load_weights(early_stopping_checkpoint)
                    break

            # update progress bar
            t.set_description("Current Loss: train = %.3g, validation = %.3g)" % (train_loss, valid_loss))
            t.refresh() 

        return train_losses, valid_losses