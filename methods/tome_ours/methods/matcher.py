import torch
import torch.nn.functional as F


def match_sinkhorn(X, Y, max_iter=3, epsilon=0.1):
    B, N, C = X.shape
    Xn = F.normalize(X, dim=-1)
    Yn = F.normalize(Y, dim=-1)
    S = torch.bmm(Xn, Yn.transpose(1, 2))  # [B, N, N]  ← 배치별 코사인 유사도

    P = (S / epsilon).softmax(-1)          # 초기 확률
    for _ in range(max_iter):
        P = P / (P.sum(dim=2, keepdim=True) + 1e-12)
        P = P / (P.sum(dim=1, keepdim=True) + 1e-12)
    return P.argmax(dim=2)                  # [B, N]


def match_greedy(X, Y):
    """
    Greedy 매칭을 위한 함수.
    X, Y는 [B, N, C] 크기의 텐서입니다.
    가장 유사한 토큰끼리 1:1 매칭을 수행합니다.

    :param X: 첫 번째 집합의 특징 (배치 크기 B, N 토큰, C 차원)
    :param Y: 두 번째 집합의 특징 (배치 크기 B, N 토큰, C 차원)
    :return: 매칭 인덱스 [B, N]
    """
    # 코사인 유사도 계산 (X, Y 사이의 내적)
    sim_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=-1)  # [B, N, N]

    # Greedy 매칭: 각 토큰에 대해 가장 유사한 인덱스를 선택
    match_indices = sim_matrix.argmax(dim=2)  # 가장 큰 유사도를 가진 인덱스 선택

    return match_indices
