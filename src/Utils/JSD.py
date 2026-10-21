#!/usr/bin/env python3
# Initial implementation by Yamil Vindas


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance


class JSDSquared(torch.nn.Module):
    """
        Computes the SQUARE of the Jensen-Shannon divergence.
        Function inspired from the discussion in: https://discuss.pytorch.org/t/jensen-shannon-divergence/2626
        # WARNING: SOFTMAX IS APPLIED INSIDE THE FUNCTION IF ASKED
        # WARNING: THIS WILL NOT WORK AS EXPECTED IF THE INPUT IS THE OUTPUT OF A LOG_SOFTMAX LAYER !!!
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, apply_softmax_input=True) -> None:
        super(JSDSquared, self).__init__()
        # We use batchmean for reduction as in Pytorch site it is stated that
        # this reduction is the one coherent with the theoretical definition
        # of th KL divergence
        # From Pythorch website on log_target: "To avoid underflow issues when
        #computing this quantity, this loss expects the argument input in the
        #log-space. The argument target may also be provided in the log-space if log_target= True."
        self.apply_softmax_input = apply_softmax_input
        self.KL = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jensen-Shannon divergence between two distributions.

        Args:
            input (torch.Tensor): Input tensor with size [Batch, C] where C is the number of classes.
            tagett (torch.Tensor): Target tensor with size [Batch] where each value is 0 <= target[i] <= C-1 or [Batch, C] with soft annotations.

        Returns:
            torch.Tensor: _description_
        """
        # Resizing the input
        if (self.apply_softmax_input):
            input = F.softmax(input)
        if (input.nelement() / target.nelement() != 1):
            # Resizing the target with hot encoding
            target = F.one_hot(target, num_classes=input.size(-1))
        input, target = input.view(-1, input.size(-1)), target.view(-1, target.size(-1))
        # The KL divergence expect the 'input' in the log-space (i.e. log of the probabilities)
        M = torch.clamp(0.5*(input + target), 1e-7, 1).log()
        input_log_space = torch.clamp(input, 1e-7, 1).log()
        target_log_space = torch.clamp(target, 1e-7, 1).log()
        # WARNING: theoric_KL(a, b) = nn.KL(b, a)
        jsd = 0.5*self.KL(M, input_log_space) + 0.5*self.KL(M, target_log_space) # theoric_KL(a, b) = nn.KL(b, a)

        return jsd

# The next functions are useful to test and validate the pytorch implementation of JSD
def KL(p, q):
  # Kullback-Leibler "from q to p"
  # p and q are np array prob distributions
  n = len(p)
  sum = 0.0
  for i in range(n):
    sum += p[i] * np.log(p[i] / q[i])
  return sum

def JS(p, q):
  m = 0.5 * (p + q)  # avg of P and Q
  left = KL(p, m)
  right = KL(q, m)
  return np.sqrt((left + right) / 2)


def test():
    """
        Test for the JSD pytorch implementation
    """
    # Creation of the JSD loss
    jsd_loss = JSDSquared(apply_softmax_input=False)

    # Test 1: P=Q
    print("\n\n=======> Test 1 <=======")
    p = torch.tensor([[1.0, 0., 0.], [0., 1.0, 0.], [0., 1.0, 0.]])
    q = torch.tensor([[1.0, 0., 0.], [0., 1.0, 0.], [0., 1.0, 0.]])
    jsd_loss_result = jsd_loss(p, q)
    print('JSD**2 for P = Q: ', jsd_loss_result, jsd_loss.KL(q, p))
    print('First test passed: ', jsd_loss_result == torch.tensor(0.))

    # Test 2: from https://jamesmccaffrey.wordpress.com/2021/07/13/jensen-shannon-distance-example/
    print("\n\n=======> Test 2 <=======")
    p = np.array([0.36, 0.48, 0.16], dtype=np.float32)
    q = np.array([0.30, 0.50, 0.20], dtype=np.float32)
    m = 0.5*(p+q)
    js_pq = JS(p, q)
    js_qp = JS(q, p)
    print('KL(P, Q) Manual: ', KL(p, q))
    print('KL(Q, P) Manual: ', KL(q, p))
    # print('KL(P, M) Manual: ', KL(p, m))
    # print('KL(Q, M) Manual: ', KL(q, m))
    print('JSD**2 Manual: ', js_pq**2, js_qp**2)
    js_sp = distance.jensenshannon(p, q)
    print('JSD**2 Scipy: ', js_sp**2)
    p_torch = torch.tensor([[0.36, 0.48, 0.16]])
    q_torch = torch.tensor([[0.30, 0.50, 0.20]])
    js_pq_torch = jsd_loss(p_torch, q_torch)
    js_qp_torch = jsd_loss(q_torch, p_torch)
    print('KL(P, Q) Ptorch: ', jsd_loss.KL(q_torch.log(), p_torch.log())) # theoric_KL(a, b) = nn.KL(b, a)
    print('KL(Q, P) Ptorch: ', jsd_loss.KL(p_torch.log(), q_torch.log())) # theoric_KL(a, b) = nn.KL(b, a)
    print('JSD Pytorch (JSD(P, Q) and JSD(Q, P)): ', js_pq_torch, js_qp_torch)



#==============================================================================#
#==============================================================================#
#==============================================================================#
def main():
    test()


if __name__=='__main__':
    main()
