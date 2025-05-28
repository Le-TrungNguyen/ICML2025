import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn
from ..compression.hosvd_var import hosvd_var

###### HOSVD base on explained variance threshold #############
class Conv2d_HOSVD_var_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var, k_hosvd = args

        # Perform convolution
        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)

        # Perform HOSVD decomposition on the input tensor
        S, u_list = hosvd_var(input, var=var)
        u0, u1, u2, u3 = u_list # B, C, H, W

        if k_hosvd is not None:
            for idx in range(4):
                k_hosvd[idx].append(u_list[idx].shape[1])
            k_hosvd[4].append(input.shape)
            k_hosvd[5].append(output.shape)

        # Save tensors for backward pass
        ctx.save_for_backward(S, u0, u1, u2, u3, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # Retrieve saved tensors
        S, u0, u1, u2, u3, weight, bias  = ctx.saved_tensors
        B, C, H, W = u0.shape[0], u1.shape[0], u2.shape[0], u3.shape[0]
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs
        
        # Compute gradient with respect to the input
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input((B,C,H,W), weight, grad_output, stride, padding, dilation, groups)

        # Compute gradient with respect to the weights
        if ctx.needs_input_grad[1]:
            _, _, K_H, K_W = weight.shape # Shape: (C', C, K_H, K_W)
            _, C_prime, H_prime, W_prime = grad_output.shape # Shape: (B, C', H', W')
            
            # Pad the input
            u2_padded = pad(u2, (0, 0, padding[0], padding[0])) # Shape: (H_padded, K2)
            u3_padded = pad(u3, (0, 0, padding[0], padding[0])) # Shape: (W_padded, K3)
            # Calculate Z1: (conv2d 1x1):
            Z1 = th.einsum("bk,bchw->kchw", u0, grad_output) # Shape: (B, K0) einsum with (B, C', H', W') -> (B, K0, C', H', W') -> (K0, C', H', W')
            #______________________________________________________________________________________________________________
            # Calculate Z2: (conv2d 1x1):
            Z2 = th.einsum("abcd,hc->abhd", S, u2_padded) # Shape: (K0, K1, K2, K3) einsum with (H_padded, K2) -> (K0, K1, H_padded, K2, K3) -> (K0, K1, H_padded, K3)
            #______________________________________________________________________________________________________________
            # Calculate Z3: (conv2d 1x1):
            Z3 = th.einsum("abhd,wd->abhw", Z2, u3_padded) # Shape: (K0, K1, H_padded, K3) einsum with (W_padded, K3) -> (K0, K1, H_padded, W_padded, K3) -> (K0, K1, H_padded, W_padded)
            # ______________________________________________________________________________________________________________
            # Calculate Z4: (conv2d H'xW'):
            if stride == dilation:
                Z4 = conv2d(Z3.permute(1, 0, 2, 3), Z1.permute(1, 0, 2, 3)).permute(1, 0, 2, 3) # Shape: (K1, K0, H_padded, W_padded) conv with (C', K0, H', W') --> (K1, C', K_H, K_W) -> (C', K1, K_H, K_W)
            else:
                Z4 = nn.grad.conv2d_weight(Z3, (C_prime, u1.shape[1], K_H, K_W), Z1, stride=stride, dilation=dilation, groups=1) # Shape (C', K1, K_H, K_W)
            #______________________________________________________________________________________________________________
            # calculate grad_weight
            if groups == C == C_prime: # Depthwise
                grad_weight = th.einsum("ckhw,ck->ckhw", Z4, u1).sum(dim=1, keepdim=True) # Shape: (C', 1, K_H, K_W)
            elif groups == 1:
                grad_weight = conv2d(Z4, u1.unsqueeze(-1).unsqueeze(-1)) # Shape: (C', K1, K_H, K_W) conv with (C, K1, 1, 1) -> (C', C, K_H, K_W)
            else:
                pass

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class Conv2d_HOSVD_var(nn.Conv2d):
    """
    Custom Conv2D layer with HOSVD4-based decomposition.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            explained_var=1,
            k_hosvd=None
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        super(Conv2d_HOSVD_var, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.explained_var = explained_var
        self.k_hosvd = k_hosvd

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.activate and th.is_grad_enabled(): # Training mode
            y = Conv2d_HOSVD_var_op.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.explained_var, self.k_hosvd)
        else: # activate is False or Inference mode
            y = super().forward(x)
        return y

def wrap_convHOSVD_var(conv, active, explained_var, k_hosvd):
    new_conv = Conv2d_HOSVD_var(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         explained_var=explained_var,
                         k_hosvd=k_hosvd
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv