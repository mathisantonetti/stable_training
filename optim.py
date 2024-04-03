import torch
import numpy as np
from sklearn.metrics import accuracy_score
import copy

def RFF_sample(n, shape):
    w = torch.randn(n)
    phi = 2*np.pi*torch.rand(n)
    if(shape == 2):
        return lambda x:np.sqrt(2)*torch.cos(w[:,None]*x[None,:]+phi[:,None])
    else:
        return lambda x:np.sqrt(2)*torch.cos(w[:,None,None]*x[None,:,:]+phi[:,None,None])

def correlation_tensor(weights, X, n_RFF=5, u=None, v=None):
    n_samples = X.shape[0]
    if(u is None):
        u = RFF_sample(n_RFF, len(X.shape))
    if(v is None):
        v = RFF_sample(n_RFF, len(X.shape))

    U, V = u(X), v(X)
    #print(U.shape, V.shape, weights.shape)
    U, V = weights[None, :, None]*U - torch.mean(weights[None, :, None]*U, axis=1)[:, None, :], weights[None, :, None]*V - torch.mean(weights[None, :, None]*V, axis=1)[:, None, :]

    print()
    return torch.sum(U[None, :, :, :, None]*V[:, None, :, None, :], axis=2)/(n_samples-1)

def F_norm(X):
    return torch.sum(X**2) - torch.sum(torch.diagonal(X, dim1=2, dim2=3)**2)


class StableOptimizer:
    def __init__(self, step=None, stable_step=None, weight_decay=5e-2, penalization_rate=1.0):
        self.step = step
        self.stable_step = stable_step
        self.weight_decay = weight_decay
        self.penalization_rate = penalization_rate
        self.hyperopt = False
        if(step is None or stable_step is None):
            self.hyperopt = True

    def train(self, model, criterion, train_data, train_score, epochs, valid_data=None, valid_score=None, val_check=1, record_function=min):
        if(self.hyperopt):
            print("Not yet implemented")
            return None
        else:
            weights = torch.ones(train_data.shape[0])
            optimizer = torch.optim.Adam(model.parameters(), lr=self.step, weight_decay=self.weight_decay)
            stable_optimizer = torch.optim.Adam([weights], lr=self.stable_step, weight_decay=self.weight_decay)

            torch.set_flush_denormal(True)
            model.train()

            u, v = RFF_sample(5,3), RFF_sample(5,3)

            train_accuracies = []
            val_accuracies = []
            weight_constraints = []
            best_state, best_val = None, 0.0
            for epoch in range(epochs):  # loop over the dataset multiple times
                # Iterate over batches and perform optimizer step on the model.
                weights.requires_grad_(False)
                model.activate(True)
                optimizer.zero_grad()
                y_pred = model(train_data)
                loss = criterion(y_pred, train_score)
                loss = (loss*weights).mean()
                loss.backward()
                optimizer.step()

                # perform optimizer step on the sample weights
                weights.requires_grad_(True)
                model.activate(False)

                stable_optimizer.zero_grad()
                F_loss = F_norm(correlation_tensor(weights, model.phi(train_data), u=u, v=v)) + self.penalization_rate/weights.mean()
                F_loss.backward()
                stable_optimizer.step()
                weight_constraints.append(weights.mean().item())

                train_acc, val_acc = accuracy_score(torch.argmax(y_pred, axis=1), torch.argmax(train_score, axis=1)), None
                if(epoch % val_check == 0):
                    hat_y_val = model(valid_data)
                    #val_loss = criterion(hat_y_val, y_val)
                    val_acc = accuracy_score(torch.argmax(hat_y_val, axis=1), torch.argmax(valid_score, axis=1))
                    if(record_function(train_acc, val_acc) > best_val):
                        best_val = record_function(train_acc, val_acc)
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    val_acc = val_accuracies[-1]

                val_accuracies.append(val_acc)
                train_accuracies.append(train_acc)
                print(f"Epoch {epoch} Loss {loss.detach().cpu().numpy():.2f} | Train Accuracy {train_acc} | Val Accuracy {val_acc}")

            return best_state, best_val, train_accuracies, val_accuracies, weight_constraints