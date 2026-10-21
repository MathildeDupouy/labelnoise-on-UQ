#!/usr/bin/env python3
# Initial implementation by Yamil Vindas

"""
    Implementation of Geometric JSD, which is partially robust against noisy
    labels (as it has one terms robust against noisy labels)
"""

import numpy as np
import torch
import torch.nn.functional as F


class AlternativeCrossEntropy(torch.nn.Module):
    """
        Alternative implementation of Cross Entropy allowing the opportunity of
        NOT USING Softmax IF ASKED.
    """
    def __init__(self, apply_softmax_p, apply_softmax_q, num_classes, device=torch.device('cpu'), class_weights=None):
        super(AlternativeCrossEntropy, self).__init__()
        self.device = device
        self.apply_softmax_p = apply_softmax_p
        self.apply_softmax_q = apply_softmax_q
        self.class_weights = class_weights
        self.num_classes = num_classes
        if (self.class_weights is None):
            self.cross_entropy = torch.nn.CrossEntropyLoss()
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, p, q):
        # Putting under the probability scores form
        if (len(p.shape) == 1):
            p = torch.nn.functional.one_hot(p, self.num_classes).float().to(self.device)
        else:
            p
        if (len(q.shape) == 1):
            q = torch.nn.functional.one_hot(q, self.num_classes).float().to(self.device)
        else:
            q
        # If softmax need to be applied
        if (self.apply_softmax_p):
            p = F.softmax(p, dim=1)
        if (self.apply_softmax_q):
            q = F.softmax(q, dim=1)



        # Computing the ouput
        p = torch.clamp(p, min=1e-7, max=1.0)
        q = torch.clamp(q, min=1e-7, max=1.0)
        ce = p.to(self.device) * torch.log(q.to(self.device))
        # Applying class weights
        if (self.class_weights is not None):
            if (type(self.class_weights) != torch.Tensor):
                self.class_weights = torch.tensor(self.class_weights)
            self.class_weights = self.class_weights.to(self.device)
            ce = self.class_weights*ce
        ce = -1*torch.sum(ce, dim=1)

        # Loss
        loss = ce.mean()

        return loss

class GeoJSDLoss(torch.nn.Module):
    def __init__(self, apply_softmax_pred, num_classes, alpha=0.5, beta=1, gamma=1, device=torch.device('cpu'), class_weights=None):
        super(GeoJSDLoss, self).__init__()
        # Some attributes
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.apply_softmax_pred = apply_softmax_pred
        self.num_classes = num_classes
        self.class_weights = class_weights

        # Cross Entropy loss functions
        self.CE_NoSoftMax = AlternativeCrossEntropy(
                                                apply_softmax_p=False,
                                                apply_softmax_q=False,
                                                num_classes=self.num_classes,
                                                device=self.device,
                                                class_weights=self.class_weights)
        self.CE_SoftMax_P = AlternativeCrossEntropy(
                                                apply_softmax_p=True,
                                                apply_softmax_q=False,
                                                num_classes=self.num_classes,
                                                device=self.device,
                                                class_weights=self.class_weights)
        self.CE_SoftMax_Q = AlternativeCrossEntropy(
                                                apply_softmax_p=False,
                                                apply_softmax_q=True,
                                                num_classes=self.num_classes,
                                                device=self.device,
                                                class_weights=self.class_weights)
        self.CE_SoftMax_P_and_Q = AlternativeCrossEntropy(
                                                apply_softmax_p=True,
                                                apply_softmax_q=True,
                                                num_classes=self.num_classes,
                                                device=self.device,
                                                class_weights=self.class_weights)

    def forward(self, pred, labels):
        # Noise-robust terrm
        if (self.apply_softmax_pred):
            l_RGJSD = self.alpha*(self.alpha-1)*(self.CE_NoSoftMax(labels, labels) - self.CE_SoftMax_P(pred, labels))
        else:
            l_RGJSD = self.alpha*(self.alpha-1)*(self.CE_NoSoftMax(labels, labels) - self.CE_NoSoftMax(pred, labels))

        # Non-robust term
        if (self.apply_softmax_pred):
            l_NGJSD = self.alpha*(self.alpha-1)*(self.CE_SoftMax_P_and_Q(pred, pred) - self.CE_SoftMax_Q(labels, pred))
        else:
            l_NGJSD = self.alpha*(self.alpha-1)*(self.CE_NoSoftMax(pred, pred) - self.CE_NoSoftMax(labels, pred))

        # Loss
        loss = self.beta * l_NGJSD + self.gamma * l_RGJSD

        return loss

# The next functions are useful to test and validate the pytorch implementation of GeoJSD
def KL(p, q):
  # Kullback-Leibler "from q to p"
  # p and q are np array prob distributions
  n = len(p)
  sum = 0.0
  for i in range(n):
    sum += p[i] * np.log(p[i] / q[i])
  return sum

def main():
    # Doing some tests
    num_classes = 3
    #==========================================================================#
    # Test 1: Alternative CE
    alternative_ce = AlternativeCrossEntropy(
                                                apply_softmax_p=False,
                                                apply_softmax_q=False,
                                                num_classes=num_classes,
                                                device=torch.device('cpu'),
                                                class_weights=None,
                                            )
    input = torch.tensor([[0.1, 0.4, 0.5]]).double()
    expected_output = -(input[0][0]*np.log(input[0][0]) + input[0][1]*np.log(input[0][1]) +  input[0][2]*np.log(input[0][2]))
    computed_output = alternative_ce(input, input)
    print("=======> TEST 1 <=======\n")
    print("Expected output: {}".format(expected_output))
    print("Computed output: {}".format(computed_output))
    print("Equal outputs: {}\n\n".format((expected_output - computed_output) < 10**(-7)))
    assert (expected_output - computed_output) < 10**(-7)
    #==========================================================================#
    # Test 2: Alternative CE
    alternative_ce = AlternativeCrossEntropy(
                                                apply_softmax_p=False,
                                                apply_softmax_q=False,
                                                num_classes=num_classes,
                                                device=torch.device('cpu'),
                                                class_weights=None,
                                            )
    input_pred = torch.tensor([[0.1, 0.4, 0.5]]).double()
    input_labels = torch.tensor([[0.1, 0.8, 0.1]]).double()
    expected_output = -(input_pred[0][0]*np.log(input_labels[0][0]) + input_pred[0][1]*np.log(input_labels[0][1]) +  input_pred[0][2]*np.log(input_labels[0][2]))
    computed_output = alternative_ce(input_pred, input_labels)
    print("=======> TEST 2 <=======\n")
    print("Expected output: {}".format(expected_output))
    print("Computed output: {}".format(computed_output))
    print("Equal outputs: {}\n\n".format((expected_output - computed_output) < 10**(-7)))
    assert (expected_output - computed_output) < 10**(-7)
    #==========================================================================#
    # Test 3: Alternative CE
    alternative_ce = AlternativeCrossEntropy(
                                                apply_softmax_p=False,
                                                apply_softmax_q=True,
                                                num_classes=num_classes,
                                                device=torch.device('cpu'),
                                                class_weights=None,
                                            )
    torch_ce = torch.nn.CrossEntropyLoss()
    input_labels = torch.tensor([[0.1, 1, 0.1]]).double()
    input_pred = torch.tensor([[0.1, 0.4, 0.5]]).double()
    expected_output = torch_ce(input_pred, input_labels) # KEEP THIS WAY!!! In torch, the first argument is the prediction and the second the ground truth
    computed_output = alternative_ce(input_labels, input_pred) # KEEP THIS WAY!!! Our implementation takes the classic def. of CE which is DIFFERENT from Torch.
    print("=======> TEST 3 <=======\n")
    print("Expected output: {}".format(expected_output))
    print("Computed output: {}".format(computed_output))
    print("Equal outputs: {}\n\n".format((expected_output - computed_output) < 10**(-7)))
    assert (expected_output - computed_output) < 10**(-7)
    #==========================================================================#
    # Test 4: Alternative CE when having 0 probabilities in the true labels !!!
    alternative_ce = AlternativeCrossEntropy(
                                                apply_softmax_p=False,
                                                apply_softmax_q=True,
                                                num_classes=num_classes,
                                                device=torch.device('cpu'),
                                                class_weights=None,
                                            )
    torch_ce = torch.nn.CrossEntropyLoss()
    input_pred = torch.tensor([[0.1, 0.4, 0.5]]).double()
    input_labels = torch.tensor([[0.0, 1, 0.0]]).double()
    expected_output = torch_ce(input_pred, input_labels) # KEEP THIS WAY!!! In torch, the first argument is the prediction and the second the ground truth
    computed_output = alternative_ce(input_labels, input_pred) # KEEP THIS WAY!!! Our implementation takes the classic def. of CE which is DIFFERENT from Torch.
    print("=======> TEST 4 <=======\n")
    print("Expected output: {}".format(expected_output))
    print("Computed output: {}".format(computed_output))
    print("Equal outputs: {}\n\n".format((expected_output - computed_output) < 10**(-7)))
    assert (expected_output - computed_output) < 10**(-7)
    #==========================================================================#
    # Test 5: Alternative CE CLASS WEIGHTS
    class_weights = torch.tensor([1/4, 1/2, 1/4])
    alternative_ce = AlternativeCrossEntropy(
                                                apply_softmax_p=False,
                                                apply_softmax_q=True,
                                                num_classes=num_classes,
                                                device=torch.device('cpu'),
                                                class_weights=class_weights,
                                            )
    torch_ce = torch.nn.CrossEntropyLoss(weight=class_weights)
    input_labels = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]).double()
    input_pred = torch.tensor([[0.1, 0.4, 0.5], [0.2, 0.3, 0.5]]).double()
    expected_output = torch_ce(input_pred, input_labels) # KEEP THIS WAY!!! In torch, the first argument is the prediction and the second the ground truth
    computed_output = alternative_ce(input_labels, input_pred) # KEEP THIS WAY!!! Our implementation takes the classic def. of CE which is DIFFERENT from Torch.
    print("=======> TEST 5 <=======\n")
    print("Expected output: {}".format(expected_output))
    print("Computed output: {}".format(computed_output))
    print("Equal outputs: {}\n\n".format((expected_output - computed_output) < 10**(-7)))
    assert (expected_output - computed_output) < 10**(-7)
    #==========================================================================#
    # Test 6: Geo JSD WITH application of softmax to pred
    alpha = 0.5
    geo_JSD_loss_hard = GeoJSDLoss(
                                    apply_softmax_pred=False,
                                    num_classes=num_classes,
                                    alpha=alpha,
                                    beta=1,
                                    gamma=1,
                                    device=torch.device('cpu'),
                                    class_weights=None
                                )
    input_pred = torch.tensor([[0.1, 0.3, 0.6]]).double()
    input_labels = torch.tensor([[0.1, 0.8, 0.1]]).double()
    geo_mean = (input_pred**(1-alpha)) * (input_labels**alpha)
    expected_output = (1-alpha)*(KL(input_pred[0], geo_mean[0]).sum()) + alpha*(KL(input_labels[0], geo_mean[0]).sum())
    computed_output = geo_JSD_loss_hard(input_pred, input_labels)
    print("=======> TEST 6 <=======\n")
    print("Expected output: {}".format(expected_output))
    print("Computed output: {}".format(computed_output))
    print("Equal outputs: {}\n\n".format((expected_output - computed_output) < 10**(-7)))
    assert (expected_output - computed_output) < 10**(-7)
    #==========================================================================#
    # Test 7: Geo JSD WITHOUT applying softmax to pred
    alpha = 0.5
    geo_JSD_loss_hard = GeoJSDLoss(
                                    apply_softmax_pred=True,
                                    num_classes=num_classes,
                                    alpha=alpha,
                                    beta=1,
                                    gamma=1,
                                    device=torch.device('cpu'),
                                    class_weights=None
                                )
    input_pred = torch.tensor([[-2, 1, 3]]).double()
    input_pred_softmax = F.softmax(input_pred, dim=1) # IMPORTANT AS THE IMPLEMENTATION OF CROSS ENTROPY OF PYTORCH APPLIED SOFTMAX TO THE INPUT !!!!!!!
    input_labels = torch.tensor([[0.1, 0.8, 0.1]]).double()
    geo_mean = (input_pred_softmax**(1-alpha)) * (input_labels**alpha)
    expected_output = (1-alpha)*KL(input_pred_softmax[0], geo_mean[0]).sum() + alpha*KL(input_labels[0], geo_mean[0]).sum()
    computed_output = geo_JSD_loss_hard(input_pred, input_labels)
    print("=======> TEST 7 <=======\n")
    print("Expected output: {}".format(expected_output))
    print("Computed output: {}".format(computed_output))
    print("Equal outputs: {}\n\n".format((expected_output - computed_output) < 10**(-7)))
    assert (expected_output - computed_output) < 10**(-7)

if __name__=='__main__':
    main()
