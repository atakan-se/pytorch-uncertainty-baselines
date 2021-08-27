import torch

def train_epoch(model, optimizer, data_loader, loss_fn, device):
    model.train()
    for data in data_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.squeeze().to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

def predict_epoch(model, data_loader, device, return_logits=False):
# If return_logits==True, returns logits and labels in two Tensors (for ECE)
    epoch_size = len(data_loader.dataset)
    model.eval()
    if return_logits:
        logit_lst = []
        label_lst = []
    correct_preds = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)
            logits = model(inputs)
            if return_logits:
                logit_lst.append(logits)
                label_lst.append(labels)
            predicted = torch.argmax(logits.data, dim=1)
            correct_preds += (predicted == labels).sum().item()
    
    if return_logits: return torch.cat(logit_lst,dim=0), torch.cat(label_lst,dim=0)
    else: return correct_preds / epoch_size


# Taken from: https://github.com/gpleiss/temperature_scaling/
class ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece