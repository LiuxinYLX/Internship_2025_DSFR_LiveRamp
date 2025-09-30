# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-09-23

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config.configuration import ModelConfig
from modules.nlp import NLPHierarchyCorrector, NLPErrorPredictor

def train_model(
        cfg: ModelConfig,
        dataset: Dataset, 
        input_dim: int,
        n_classes_per_level: list,
        model_read_name: str,
        model_save_name: str,
        load_from_checkpoint: bool = False,
        return_levels: bool = False,
    ) -> nn.Module:

    if return_levels:
        model = NLPHierarchyCorrector(
            input_dim = input_dim,
            hidden_dim = cfg.hidden_dim,
            n_classes_per_level = n_classes_per_level,
        ).to(cfg.device)
    else:
        model = NLPErrorPredictor(
            input_dim = input_dim,
            hidden_dim = cfg.hidden_dim
        ).to(cfg.device)

    if load_from_checkpoint:
        model_read_path = os.path.join(cfg.model_save_path, model_read_name)
        model.load_state_dict(torch.load(model_read_path))
        model = model.to(cfg.device)

    # -------- 1) 构建 DataLoader --------
    dataLoader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # -------- 2) 预计算 is_error 的类别权重 --------
    all_lb = []
    for batch in dataLoader:
        if return_levels:
            _, _, _, lb = batch
        else:
            _, lb = batch
        all_lb.append(lb.view(-1).long())
    all_lb = torch.cat(all_lb)

    counts = torch.bincount(all_lb, minlength=2).float()
    # weight_c = N / (K * count_c)
    class_weights = counts.sum() / (2.0 * counts.clamp(min=1.0))
    class_weights = class_weights.to(cfg.device)
    
    #coef_error = 5

    # -------- 3) 定义损失函数 --------
    loss_fn_level = nn.CrossEntropyLoss()
    loss_fn_label = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # -------- 4) 训练模型 --------
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in dataLoader:
            if return_levels:
                xb, yb, ylb, lb = batch
            else:
                xb, lb = batch

            xb = xb.to(cfg.device)
            lb = lb.to(cfg.device)
            preds = model(xb)
            # X, y, ylabels, label
            if return_levels:
                yb = yb.to(cfg.device)
                ylb = ylb.to(cfg.device)
                is_error = lb.view(-1).bool()
                level_loss = 0.0
                for i in range(len(preds)-1):
                    if (~is_error).any():
                        level_loss += loss_fn_level(preds[i][~is_error], yb[:,i][~is_error])
                    if (is_error).any():
                        level_loss += loss_fn_level(preds[i][is_error], ylb[:,i][is_error])
                label_loss = loss_fn_label(preds[-1], lb.long().squeeze(-1))
                loss = level_loss + label_loss
            # X, label
            else:
                label_loss = loss_fn_label(preds[-1], lb.long().squeeze(-1))
                loss = label_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{cfg.epochs}, Loss: {total_loss:.4f}")
    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.model_save_path, model_save_name))

def inference_model(
        cfg: ModelConfig,
        dataset: Dataset,
        input_dim: int,
        n_classes_per_level: list,
        model_read_name: str,
        return_levels: bool = True,
    ) -> tuple[float, np.ndarray]:
    
    if return_levels:
        model = NLPHierarchyCorrector(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            n_classes_per_level=n_classes_per_level
        )
    else:
        model = NLPErrorPredictor(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim
        )

    model_read_path = os.path.join(cfg.model_save_path, model_read_name)
    model.load_state_dict(torch.load(model_read_path, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    dataLoader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        if return_levels:
            print("Inference: return_levels=True, will predict ylabels and label.")
            all_labels = []
            all_probs = []

            all_yb = []
            all_ylb = []
            for batch in dataLoader:
                xb, yb, ylb, lb = batch
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                ylb = ylb.to(cfg.device)
                lb = lb.to(cfg.device)

                preds = model(xb)
                all_labels.append(lb.cpu().numpy())

                # Classe prédite pour chaque niveau et is_error
                batch_preds = [torch.argmax(pred, dim=1).cpu().numpy() for pred in preds]
                batch_preds = np.stack(batch_preds, axis=1)  # (batch_size, n_levels+1)
                predictions.append(batch_preds)
                
                # Probabilities entre 0 et 1 pour is_error
                logits = preds[-1]
                probs = torch.softmax(logits, dim=1)[:,1]
                all_probs.append(probs.cpu().numpy())

                all_yb.append(yb.cpu().numpy())
                all_ylb.append(ylb.cpu().numpy())

            pred_labels = np.vstack(predictions)
            true_labels = np.concatenate(all_labels, axis=0)
            pred_probs = np.concatenate(all_probs, axis=0)
            # 评估 is_error
            pred_is_error = pred_labels[:,-1]
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                acc = accuracy_score(true_labels, pred_is_error)
                precision = precision_score(true_labels, pred_is_error)
                recall = recall_score(true_labels, pred_is_error)
                f1 = f1_score(true_labels, pred_is_error)
                cm = confusion_matrix(true_labels, pred_is_error)
                print(f"[is_error] Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
                print(f"[is_error] Confusion Matrix:\n{cm}")
                # 评估每一级等级预测
                all_yb = np.concatenate(all_yb, axis=0)
                all_ylb = np.concatenate(all_ylb, axis=0)
                for i in range(pred_labels.shape[1]-1):
                    # 只对非错误样本评估等级预测
                    mask = (true_labels == 0)
                    if mask.sum() == 0:
                        print(f"[level {i+1}] No correct samples to evaluate.")
                        continue
                    acc_lvl = accuracy_score(all_yb[mask,i], pred_labels[mask,i])
                    print(f"[level {i+1}] Accuracy: {acc_lvl:.4f}")
            except ImportError:
                print("scikit-learn is not installed, cannot compute metrics.")
        else:
            print("Inference: return_levels=False, only predict label (is_error).")
            all_labels = []
            all_probs = []
            for batch in dataLoader:
                xb, lb = batch
                xb = xb.to(cfg.device)
                lb = lb.to(cfg.device)
                preds = model(xb)
                all_labels.append(lb.cpu().numpy())

                # Classe prédite (0 ou 1)
                pred_is_error = torch.argmax(preds[0], dim=1).cpu().numpy()
                predictions.append(pred_is_error)

                # Probabilities entre 0 et 1
                logits = preds[-1]
                probs = torch.softmax(logits, dim=1)[:,1]
                all_probs.append(probs.cpu().numpy())
                
            pred_labels = np.concatenate(predictions, axis=0)
            true_labels = np.concatenate(all_labels, axis=0)
            pred_probs = np.concatenate(all_probs, axis=0)
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                acc = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels)
                recall = recall_score(true_labels, pred_labels)
                f1 = f1_score(true_labels, pred_labels)
                cm = confusion_matrix(true_labels, pred_labels)
                print(f"[is_error] Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
                print(f"[is_error] Confusion Matrix:\n{cm}")
            except ImportError:
                print("scikit-learn is not installed, cannot compute metrics.")
    return pred_labels, pred_probs



def decode_predictions(
    pred_labels: np.ndarray,
    reverse_label_map: dict[str, dict[int, str]],
    hierarchy_cols: list[str],
    df: pd.DataFrame = None
) -> pd.DataFrame:
    
    decoded_preds = {
        f"pred_{col}": [reverse_label_map[col][idx] for idx in pred_labels[:, i]]
        for i, col in enumerate(hierarchy_cols)
    }

    pred_df = pd.DataFrame(decoded_preds)
    if df is not None:
        return pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    return pred_df