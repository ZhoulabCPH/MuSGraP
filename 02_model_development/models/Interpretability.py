import torch
import matplotlib.pyplot as plt


def visualize_concept_contributions(model, outputs, concept_k, topk=10):
    """
    可视化概念对预测结果的贡献，展示 WSI 和 CT 的贡献。
    """
    c_wsi = outputs["c_wsi"]  # [B, K]
    c_ct = outputs["c_ct"]  # [B, K]

    # 获取某个概念的贡献
    heatmap_wsi = c_wsi.detach().cpu().numpy()
    heatmap_ct = c_ct.detach().cpu().numpy()

    # 绘制热力图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(heatmap_wsi, cmap='hot', interpolation='nearest')
    plt.title(f"WSI Concept {concept_k} Heatmap")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_ct, cmap='hot', interpolation='nearest')
    plt.title(f"CT Concept {concept_k} Heatmap")

    plt.show()

    return heatmap_wsi, heatmap_ct


def top_patches_for_concept(A_wsi: torch.Tensor, patch_paths: list, concept_k: int, topk: int = 10):
    """
    获取与某个概念相关的 top patches（在 WSI 中贡献较大的图像块）
    """
    scores = A_wsi[:, concept_k]
    idx = torch.topk(scores, k=min(topk, scores.numel())).indices.cpu().tolist()
    return [(patch_paths[i], float(scores[i].cpu())) for i in idx]


def top_concept_pairs_for_class(model, outputs, sample_idx=0, class_idx=0, topk=10):
    """
    获取贡献最大的概念对（k_wsi, k_ct），用于解释模型如何根据这些概念组合做出决策。
    """
    c_wsi = outputs["c_wsi"][sample_idx]  # [K]
    c_ct = outputs["c_ct"][sample_idx]  # [K]
    A_class = model.head.synergy_matrix_for_class(class_idx)  # [K,K]

    contrib = torch.outer(c_wsi, c_ct) * A_class  # [K,K]
    flat = contrib.flatten()
    vals, idx = torch.topk(flat, k=min(topk, flat.numel()))
    pairs = []
    K = model.K
    for v, ind in zip(vals.cpu().tolist(), idx.cpu().tolist()):
        kw = ind // K
        kc = ind % K
        pairs.append((kw, kc, v))
    return pairs
