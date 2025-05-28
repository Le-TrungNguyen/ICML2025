import torch
import torch.nn as nn
from torch.autograd import Function

from ..compression.hosvd_var import hosvd_var
###### HOSVD based on explained variance threshold #############
class Linear_HOSVD_var_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, var, k_hosvd = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Perform decomposition
        S, U_list = hosvd_var(input, var=var)
        if k_hosvd is not None:
            # Log information for estimating activation memory
            for i, U in enumerate(U_list):
                k_hosvd[i].append(U.shape[1])
            k_hosvd[i + 1].append(input.shape)
            k_hosvd[i + 2].append(output.shape)

        # Save information for backpropagation
        ctx.save_for_backward(S, weight, bias)
        ctx.U_list = U_list
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        S, weight, bias = ctx.saved_tensors

        U_list = ctx.U_list

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if len(U_list) == 4:
                U1, U2, U3, U4 = U_list
                # grad_weight = torch.einsum('bhwc,bhwd->dc', restore_hosvd_4_mode(S, [U1, U2, U3, U4]), grad_output)
                ################ Low rank gradient calculation ################:
                Z1 = torch.einsum("Ba,BHWD->aHWD", U1, grad_output) # Shape: (B, K1) and (B, H, W, D) -> (K1, H, W, D)
                Z2 = torch.einsum("Hb,abcd->aHcd", U2, S) # Shape: (H, K2) and (K1, K2, K3, K4) -> (K1, H, K3, K4)
                Z3 = torch.einsum("Wc,aHWD->aHcD", U3, Z1) # Shape: (W, K3) and (K1, H, W, D) -> (K1, H, K3, D)
                Z4 = torch.einsum("Cd,aHcd->aHCc", U4, Z2) # Shape: (C, K4) and (K1, H, K3, K4) -> (K1, H, C, K3)
                grad_weight = torch.einsum("aHcD,aHCc->DC", Z3, Z4) # Shape: (K1, H, K3, D) and (K1, H, C, K3) -> (D, C)
            elif len(U_list) == 3:
                U1, U2, U3 = U_list
                Z1 = torch.einsum('blo,bk->lok', grad_output, U1) # Shape: B, L, O and B, K1 -> L, O, K1
                Z2 = torch.einsum('abc,lb->acl', S, U2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
                Z3 = torch.einsum('acl,ic->ail', Z2, U3) # Shape: K1, K3, L and I, K3 -> K1, I, L
                grad_weight = torch.einsum('lok,kil->oi', Z1, Z3) # Shape: L, O, K1 and K1, I, L -> O, I

            # max_abs_diff = torch.max(torch.abs(grad_weight - grad_weight_)).item()
            # print(f"Maximum absolute difference: {max_abs_diff:.6f}")

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None

class Linear_HOSVD_var(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            var=0.9,
            k_hosvd = None):
        super(Linear_HOSVD_var, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.var = var
        self.k_hosvd = k_hosvd

    def forward(self, input):
        if self.activate and torch.is_grad_enabled(): # Training mode
            output = Linear_HOSVD_var_op.apply(input, self.weight, self.bias, self.var, self.k_hosvd)
        else: # activate is False or Validation mode
            output = super().forward(input)
        return output
    

def wrap_linearHOSVD_var(linear, active, SVD_var, k_hosvd):
    has_bias = (linear.bias is not None)
    new_linear = Linear_HOSVD_var(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        var=SVD_var,
                        k_hosvd = k_hosvd
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear