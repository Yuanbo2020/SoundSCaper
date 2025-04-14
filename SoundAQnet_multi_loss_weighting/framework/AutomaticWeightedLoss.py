# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

        self.num = num

    def forward(self, input_losses, alpha_list):

        loss_list = []
        sigma_list = []
        loss_sum = 0
        for i, loss in enumerate(input_losses):
            sigma_list.append(float(self.params[i]))
            current_loss = 1 / (self.params[i] ** 2) * alpha_list[i] * loss + torch.log(1 + self.params[i])
            loss_list.append(current_loss)
            loss_sum += current_loss
        print(sigma_list)
        return loss_sum, loss_list


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())