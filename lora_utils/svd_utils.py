import torch
from typing import Tuple


def get_linear_rec_svd(
    input_matrix: torch.Tensor, 
    rank: int, 
    n_iter: int, 
    random_state: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(random_state)
    mat = input_matrix.float()
    U, S, V = torch.svd_lowrank(mat, q=rank, niter=n_iter)
    S_sqrt = torch.sqrt(S)
    # Scale columns/rows properly
    U_sigma = U * S_sqrt.unsqueeze(0)        
    sigma_V_t = S_sqrt.unsqueeze(1) * V.T    
    reconstructed_matrix = U_sigma @ sigma_V_t
    return reconstructed_matrix, U_sigma, sigma_V_t
    # mat = input_matrix.float()
    # U, S, V = torch.svd_lowrank(mat, q=min(rank, min(mat.shape)), niter=n_iter)

    # true_rank = S.shape[0]  # actual rank returned by svd_lowrank
    # pad = rank - true_rank   # how many zeros to pad

    # if pad > 0:
    #     # Pad S with zeros
    #     S = torch.cat([S, torch.zeros(pad, device=S.device)], dim=0)

    #     # Pad U with orthonormal columns (or zeros)
    #     m = U.shape[0]
    #     U_pad = torch.zeros(m, pad, device=U.device)
    #     U = torch.cat([U, U_pad], dim=1)

    #     # Pad V with orthonormal columns (or zeros)
    #     n = V.shape[0]
    #     V_pad = torch.zeros(n, pad, device=V.device)
    #     V = torch.cat([V, V_pad], dim=1)

    # # Now U, S, V all have size rank
    # U_sigma = U * S
    # V_t = V.T
    # reconstructed_matrix = U_sigma @ V_t

    # # enc_inv safely handle zeros in S
    # S_safe = S.clone()
    # S_safe[S_safe == 0] = 1.0
    # enc_inv = (1.0 / S_safe).unsqueeze(1) * U.T
    # dec_inv = V

    # return reconstructed_matrix, U_sigma, V_t, enc_inv, dec_inv


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
