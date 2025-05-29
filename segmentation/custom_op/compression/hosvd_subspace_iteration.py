import torch as th
import numpy as np

def Gram_Schmidt(matrix):
    new_matrix = matrix.clone()

    original_type = new_matrix.dtype #th.linalg.qr doesn't support helf precision types such as th.bfloat16
    new_matrix, _ = th.linalg.qr(new_matrix.to(dtype=th.float32))
    new_matrix = new_matrix.to(dtype=original_type)

    return new_matrix

def set_random(shape, random_seed=233, device='cuda'):
    th.manual_seed(np.random.RandomState(random_seed).randint(1_000_000_000))
    random_tensor = th.randn(shape, device=device)
    return random_tensor

def unfolding(n, A):
    """
    Unfold tensor A along the nth mode.
    Args:
        n (int): The mode along which to unfold.
        A (torch.Tensor): The tensor to be unfolded.
    
    Returns:
        torch.Tensor: Unfolded tensor of shape (shape[n], prod(shape) / shape[n])

    For example: 
    A with shape (a, b, c, d) and n = 0 -> unfolded_A with shape (a, b*c*d)
    A with shape (a, b, c, d) and n = 1 -> unfolded_A with shape (b, a*c*d)
    A with shape (a, b, c, d) and n = 2 -> unfolded_A with shape (c, a*b*d)
    A with shape (a, b, c, d) and n = 3 -> unfolded_A with shape (d, a*b*c)
    """
    shape = A.shape
    # Permute dimensions to bring nth dimension to the front
    sizelist = list(range(len(shape)))
    sizelist[n], sizelist[0] = 0, n
    # Reshape after permuting to get unfolded matrix
    return A.permute(sizelist).reshape(shape[n], -1)

def find_U(unfolded_tensor, previous_U, reuse_U=False, rank=1, device='cuda'):
    n, m = unfolded_tensor.shape
    rank = min(m, n, rank)

    if reuse_U:
        V = th.matmul(unfolded_tensor.t(), previous_U)
    else:
        V = set_random((m, rank))

    U = th.matmul(unfolded_tensor, V)
    U = Gram_Schmidt(U)

    return U.detach()

def find_U_mode_n(n, A, rank, reuse_U, previous_U):
    unfolded_A = unfolding(n, A)
    return find_U(unfolded_A, previous_U, reuse_U=reuse_U, rank=rank, device='cuda')

def hosvd_subspace_iteration(A, previous_Ulist, reuse_U, rank):
    S = A.clone()
    u_list = []

    if type(rank) != list: rank = [rank] * A.dim()
    # Loop over each mode of the tensor
    for i in range(A.dim()):
        if reuse_U: previous_U = previous_Ulist[i]
        else: previous_U = None
        u = find_U_mode_n(n=i, A=A, rank=rank[i], reuse_U=reuse_U, previous_U=previous_U)
        # Perform tensor contraction along the ith mode
        S = th.tensordot(S, u, dims=([0], [0]))
        u_list.append(u)
    return S, u_list

def restore_hosvd_subspace_iteration(S, u_list):
    """
    Restore the original tensor from the core tensor and factor matrices.
    
    Args:
        S (torch.Tensor): Core tensor from HOSVD.
        u_list (list): List of factor matrices from HOSVD.
    
    Returns:
        torch.Tensor: The restored tensor.
    """
    restored_tensor = S.clone()
    # Perform tensor contraction to restore the original tensor
    for u in u_list:
        restored_tensor = th.tensordot(restored_tensor, u.t(), dims=([0], [0]))

    return restored_tensor