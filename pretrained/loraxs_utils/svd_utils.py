import torch
from typing import Tuple


def get_linear_rec_svd(input_matrix: torch.Tensor, rank: int, n_iter: int,
                       random_state: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mat = input_matrix.float()
    U, S, V = torch.svd_lowrank(mat, q=rank, niter=n_iter)
    U = U[:, :rank]   
    S = S[:rank]     
    V = V[:, :rank]   
    U_sigma = U * S          
    V_t = V.T                
    reconstructed_matrix = U_sigma @ V_t
    enc_inv = (1.0 / S).unsqueeze(1) * U.T   
    dec_inv = V                                
    return reconstructed_matrix, U_sigma, V_t, enc_inv, dec_inv

#     import torch
# from typing import Tuple


# def get_linear_rec_svd(input_matrix: torch.Tensor, rank: int, n_iter: int,
#                        random_state: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     # Cast to float32 for numerical stability during SVD
#     mat = input_matrix.float()

#     # torch.svd_lowrank is the GPU-compatible equivalent of sklearn's TruncatedSVD
#     # (same randomized algorithm; random_state is not forwarded — seed externally if needed)
#     U, S, V = torch.svd_lowrank(mat, q=rank, niter=n_iter)
#     U = U[:, :rank]   # (m, rank)
#     S = S[:rank]      # (rank,)
#     V = V[:, :rank]   # (n, rank)

#     U_sigma = U * S          # (m, rank)  — left singular vectors scaled by sigma
#     V_t = V.T                # (rank, n)
#     reconstructed_matrix = U_sigma @ V_t

#     enc_inv = (1.0 / S).unsqueeze(1) * U.T   # (rank, m) — left pseudo-inverse of U_sigma
#     dec_inv = V                                # (n, rank) — right pseudo-inverse of V_t

#     return reconstructed_matrix, U_sigma, V_t, enc_inv, dec_inv