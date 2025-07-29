import torch

class BaseClosure:
    def __init__(self, optimizer, net, criterion, method, dummy_data, dummy_label, label_pred, original_dy_dx):
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.method = method
        self.dummy_data = dummy_data
        self.dummy_label = dummy_label
        self.label_pred = label_pred
        self.original_dy_dx = original_dy_dx

    def __call__(self):
        self.optimizer.zero_grad()
        pred = self.net(self.dummy_data)
        if self.method == 'DLG':
            dummy_loss = - torch.mean(
                torch.sum(torch.softmax(self.dummy_label.float(), -1) * torch.log(torch.softmax(pred, -1)),
                          dim=-1))
        elif self.method == 'iDLG':
            dummy_loss = self.criterion(pred, self.label_pred)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.parameters(), create_graph=True)
        # dummy_dy_dx = cge(f, params_dict, mask_dict, mArgs.zoo_step_size, net, dummy_data, dummy_label, F.cross_entropy)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, self.original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        return grad_diff
