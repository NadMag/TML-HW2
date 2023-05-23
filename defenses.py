import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)
                           

    # init delta (adv. perturbation) - FILL ME
    input_shape = data_tr[0][0].shape
    delta = torch.zeros((batch_size, *input_shape), device=device)

    # total number of updates - FILL ME
    outer_epochs = int(np.ceil(epochs / m))

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    model = model.to(device)

    for epoch in range(outer_epochs):
        for i, (inputs, labels) in enumerate(loader_tr, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            curr_batch_size = inputs.shape[0]

            for k in range(m):
                adv_input = inputs + delta[:curr_batch_size]
                # Paper doesnt explicitly say to clip, but I assume we do - like in PGD
                adv_input = torch.clip(adv_input, 0, 1)
                adv_input.requires_grad = True

                optimizer.zero_grad()
                outputs = model(adv_input)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                adv_grad = adv_input.grad
                delta[:curr_batch_size] = delta[:curr_batch_size] + (eps *torch.sign(adv_grad))
                delta = torch.clip(delta, -eps, eps)

                # I Assume iteration == one (adverserial) batch processing 
                iters_elapsed = (((epoch * int(np.ceil(len(data_tr)/batch_size))) + i) * m) + k
                if (iters_elapsed % scheduler_step_iters == 0):
                    lr_scheduler.step()

        #TODO: Remove!!!!        
        print(f'Epoch: {epoch}: {loss.item()}') 

    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        # x is a single instance we batch over noise
        # Bad code - we need to know the classes
        self.model.eval()
        counts = None
        with torch.no_grad():
            for i in range(0, n, batch_size):
                rep_count = min(n-i, batch_size)
                x_batch = x.repeat((rep_count, 1, 1, 1))
                noise = self.sigma * torch.randn_like(x_batch)
                x_batch = x_batch + noise
                outputs = self.model(x_batch)
                num_labels = outputs.shape[1]
                if counts is None:
                    counts = torch.zeros((num_labels))
                predictions = torch.argmax(outputs, dim=1)
                batch_counts = torch.bincount(predictions, minlength=num_labels)
                counts += batch_counts
            
            return counts
        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c) - FILL ME
        class_selection_counts = self._sample_under_noise(x, n0, batch_size)
        top_class = torch.argmax(class_selection_counts).item()
        
        # compute lower bound on p_c - FILL ME
        lb_counts = self._sample_under_noise(x, n, batch_size)
        top_class_counts = lb_counts[top_class].item()
        lower_bound = proportion_confint(top_class_counts, n, alpha=2*alpha, method="beta")[0]

        if (lower_bound <= 0.5):
            return SmoothedModel.ABSTAIN, 0

        radius = self.sigma * norm.ppf(lower_bound)
        # done
        return top_class, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        

        # done
        return mask, trigger
