from torchsummary import summary
from torch.autograd import Variable
from models.ssd import build_ssd
from models.refinedet import build_refinedet
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
opt = parser.parse_args()
# print(opt)


def print_model_parm_flops(model,size):


    multiply_adds = True  # FLOPs include multiply ops and add ops
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    # resnet = models.alexnet()
    # resnet = models.resnet152()

    model = model
    foo(model)

    input = Variable(torch.rand(3, size, size).unsqueeze(0), requires_grad=True).cuda()
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    net1 = build_ssd('train', 300, 7).to(device)
    net2 = build_refinedet('train', 320, 7).to(device)
    # summary(net1, input_size=(3,300,300))

    total1 = sum([param.nelement() for param in net1.parameters()])
    total2 = sum([param.nelement() for param in net2.parameters()])
    print('  + SSD Number of params: %.2fM' % (total1 / 1e6))
    print('  + RefineDet Number of params: %.2fM' % (total2 / 1e6))

    print_model_parm_flops(net1,300)
    print_model_parm_flops(net2, 320)