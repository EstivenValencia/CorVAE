"""
Lógica de entrenamiento y evaluación para un Autoencoder Variacional (VAE) 
con arquitectura Transformer, adaptado para datos tabulares mixtos.
Incluye un ciclo de pre-entrenamiento no supervisado y un ciclo opcional de 
fine-tuning supervisado.
"""

# --- 1. Importaciones ---
import os
import time
import json
import pickle
import argparse
import warnings
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Importaciones locales del proyecto
from utils import preprocessing, read_json_config
from models_vae import (ModelVAE, compute_loss, EncoderModel, DecoderModel, LossTracker,
                        MLP_ModelVAE, MLP_EncoderModel, MLP_DecoderModel)

warnings.filterwarnings('ignore')

# --- 2. Lógica de Entrenamiento y Validación ---

def train_vae(model, loader, optimizer, device, beta=1.0, alpha=0.0):
    """
    Ejecuta una época de entrenamiento para el modelo VAE.

    Args:
        model (nn.Module): El modelo VAE a entrenar.
        loader (DataLoader): DataLoader con los datos de entrenamiento.
        optimizer (torch.optim.Optimizer): Optimizador para actualizar los pesos.
        device (torch.device): Dispositivo (CPU o CUDA) donde se ejecuta el entrenamiento.
        beta (float): Factor de ponderación para la pérdida de divergencia KL.
        alpha (float): Factor de ponderación para la pérdida de clasificación (usado en fine-tuning).

    Returns:
        Tuple[float, ...]: Tupla con las pérdidas promedio de la época (MSE, CE, KLD, Total, Clasificación).
    """
    model.train()
    total_mse = total_ce = total_kld = total_loss = total_class_loss = 0.0
    n_samples = 0
    
    loader_tqdm = tqdm(loader, total=len(loader), desc="Training")
    for X_num_batch, X_cat_batch, y_batch in loader_tqdm:
        batch_size = X_num_batch.size(0)
        n_samples += batch_size

        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Realiza el forward pass para obtener reconstrucciones y parámetros latentes.
        recon_num, recon_cat, mu_z, logvar_z, class_logits = model(X_num_batch, X_cat_batch)

        # 2. Calcula las pérdidas de reconstrucción (MSE, CE) y regularización (KLD).
        loss_mse, loss_ce, loss_kld, _ = compute_loss(
            X_num_batch, X_cat_batch, recon_num, recon_cat, mu_z, logvar_z
        )
        loss = loss_mse + loss_ce + beta * loss_kld

        # 3. (Opcional) Añade la pérdida de clasificación si el modelo está en modo fine-tuning.
        loss_class = torch.tensor(0.0, device=device) 
        if alpha > 0 and class_logits is not None:
            loss_class = F.cross_entropy(class_logits, y_batch.long())
            loss += alpha * loss_class

        # 4. Realiza el paso de optimización (backpropagation).
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. Acumula las pérdidas para el cálculo del promedio de la época.
        total_mse  += loss_mse.item() * batch_size
        total_ce   += loss_ce.item() * batch_size
        total_kld  += loss_kld.item() * batch_size
        total_class_loss += loss_class.item() * batch_size
        total_loss += loss.item() * batch_size

    # Calcula las pérdidas promedio por muestra para toda la época.
    avg_mse = total_mse / n_samples
    avg_ce = total_ce / n_samples
    avg_kld = total_kld / n_samples
    avg_total = total_loss / n_samples
    avg_class_loss = total_class_loss / n_samples

    return avg_mse, avg_ce, avg_kld, avg_total, avg_class_loss


def validate_vae(model, loader, device, beta=1.0, alpha=0.0):
    """
    Ejecuta una época de validación para el modelo VAE.

    Args:
        (Ver función train_vae para la descripción de los argumentos).

    Returns:
        Tuple[float, ...]: Tupla con las pérdidas promedio de la época de validación.
    """
    model.eval()
    total_mse = total_ce = total_kld = total_loss = total_class_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in loader:
            batch_size = X_num_batch.size(0)
            n_samples += batch_size

            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)
            y_batch = y_batch.to(device)

            # 1. Realiza el forward pass en modo de evaluación (sin cálculo de gradientes).
            recon_num, recon_cat, mu_z, logvar_z, class_logits = model(X_num_batch, X_cat_batch)
            
            # 2. Calcula las pérdidas de la misma forma que en el entrenamiento.
            loss_mse, loss_ce, loss_kld, _ = compute_loss(
                X_num_batch, X_cat_batch, recon_num, recon_cat, mu_z, logvar_z
            )
            loss = loss_mse + loss_ce + beta * loss_kld

            loss_class = torch.tensor(0.0, device=device) 
            if alpha > 0 and class_logits is not None:
                loss_class = F.cross_entropy(class_logits, y_batch.long())
                loss += alpha * loss_class
            
            # 3. Acumula las pérdidas para el promedio final.
            total_mse  += loss_mse.item() * batch_size
            total_ce   += loss_ce.item() * batch_size
            total_kld  += loss_kld.item() * batch_size
            total_class_loss += loss_class.item() * batch_size
            total_loss += loss.item() * batch_size
            
    # Calcula las pérdidas promedio por muestra.
    avg_mse = total_mse / n_samples
    avg_ce = total_ce / n_samples
    avg_kld = total_kld / n_samples
    avg_total = total_loss / n_samples
    avg_class_loss = total_class_loss / n_samples

    return avg_mse, avg_ce, avg_kld, avg_total, avg_class_loss


def save_best_encoder_decoder(model, model_save_path, num_cols, categories, pre_encoder, pre_decoder, num_scaler, cat_encoder, label_encoder,
                               X_train_num, X_train_cat, y_train_enc, X_test_num, X_test_cat, y_test_enc, ckpt_dir, device, random_state=None):
    """
    Guarda los componentes del mejor modelo VAE y las representaciones latentes.

    Esta función carga el mejor modelo guardado, extrae los pesos del encoder y
    decoder, los guarda como modelos independientes y genera las representaciones
    latentes (embeddings) para los datos de entrenamiento y prueba.
    """
    encoder_save_path = os.path.join(ckpt_dir, f'encoder_seed_{random_state}.pt')
    decoder_save_path = os.path.join(ckpt_dir, f'decoder_seed_{random_state}.pt')

    # Carga los pesos del mejor modelo VAE que fue guardado durante el entrenamiento.
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Guarda los objetos de preprocesamiento necesarios para la reconstrucción futura.
    with open(os.path.join(ckpt_dir, f'pre_encoders_seed_{random_state}.pkl'), 'wb') as f:
        pickle.dump({
            "num_cols": num_cols, "categories": categories, 'num_scaler': num_scaler,
            'cat_ordinal_encoder': cat_encoder, 'label_encoder': label_encoder
        }, f)

    with torch.no_grad():
        # Transfiere los pesos desde el VAE completo a los modelos de Encoder y Decoder.
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        # Guarda los estados de los modelos Encoder y Decoder de forma independiente.
        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        # Genera y guarda las representaciones latentes para los conjuntos de datos por lotes (batches).
        start_time = time.time()
        
        def extract_z_in_batches(encoder, x_num, x_cat, batch_size=1024):
            z_list = []
            for i in range(0, x_num.size(0), batch_size):
                z_batch = encoder(x_num[i:i+batch_size], x_cat[i:i+batch_size])
                z_list.append(z_batch.detach().cpu().numpy())
            return np.concatenate(z_list, axis=0)

        train_z = extract_z_in_batches(pre_encoder, X_train_num, X_train_cat)
        encoder_inference_time = (time.time() - start_time) / X_train_num.size(0)
        test_z = extract_z_in_batches(pre_encoder, X_test_num, X_test_cat)

        # Guarda los embeddings y las etiquetas correspondientes.
        np.save(os.path.join(ckpt_dir, f'train_z_seed_{random_state}.npy'), train_z)
        np.save(os.path.join(ckpt_dir, f'train_y_seed_{random_state}.npy'), y_train_enc)
        np.save(os.path.join(ckpt_dir, f'test_z_seed_{random_state}.npy'), test_z)
        np.save(os.path.join(ckpt_dir, f'test_y_seed_{random_state}.npy'), y_test_enc)

        print('Encoder, Decoder y representaciones latentes guardados correctamente.')
        return train_z, encoder_inference_time


# --- 3. Orquestador Principal del Entrenamiento ---

def main(config, base_dir, set_data, encoding='ordinal', random_state=0, batch_size=64, pretrain_epochs=50, finetune_epochs=30, concat_label=False,ckpt_dir='ckpt', hyperparams = {}, architecture='transformer'):
    
    # --- 1. Configuración de Hiperparámetros ---
    # Parámetros para el pre-entrenamiento del VAE.
    max_beta = hyperparams.get('max_beta', 1e-2)
    min_beta = hyperparams.get('min_beta', 1e-5)
    lambda_ = hyperparams.get('lambda_', 0.7)
    lr_pretrain = hyperparams.get('lr_pretrain', 1e-3)
    wd_pretrain = hyperparams.get('wd_pretrain', 0)
    d_token = hyperparams.get('d_token', 4)
    token_bias = hyperparams.get('token_bias', True)
    n_head = hyperparams.get('n_head', 1)
    factor = hyperparams.get('factor', 32)
    num_layers = hyperparams.get('num_layers', 2)
    early_stop_counter_pretrain = hyperparams.get('early_stop_counter_pretrain', 30)

    # Parámetros específicos del MLP (solo usados si architecture == 'mlp').
    n_hidden_layers = hyperparams.get('n_hidden_layers', 2)
    hidden_dim = hyperparams.get('hidden_dim', 128)
    mlp_dropout = hyperparams.get('mlp_dropout', 0.3)
    
    # Parámetros para el fine-tuning supervisado.
    dropout_ft = hyperparams.get('dropout_ft', 0.3)
    alpha_ft = hyperparams.get("alpha_ft", 1e-5)
    lr_ft = hyperparams.get("lr_ft", 1e-4)
    wd_ft = hyperparams.get('wd_ft', 0)
    early_stop_counter_ft = hyperparams.get("early_stop_counter_ft", 30)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = os.path.join(ckpt_dir, f'best_vae_seed_{random_state}.pt')
    ft_model_save_path = os.path.join(ckpt_dir, f'ft_best_vae_seed_{random_state}.pt')

    # --- 2. Preparación y Carga de Datos ---
    X_train, y_train, X_test, y_test = set_data
    # Realiza el preprocesamiento de los datos (escalado, codificación).
    (X_num_train, X_cat_train, X_num_test, X_cat_test, y_train_enc, y_test_enc,
     num_scaler, cat_encoder, label_encoder) = preprocessing(
        config, X_train, y_train, X_test, y_test, encoding=encoding, 
        concat_label=concat_label, random_state=random_state
    )

    categories = [len(set(X_cat_train[:, i])) for i in range(X_cat_train.shape[1])]
    num_classes = len(set(y_train_enc))

    # Convierte los datos de numpy a tensores de PyTorch y crea los DataLoaders.
    train_ds = TensorDataset(torch.tensor(X_num_train, dtype=torch.float32), 
                             torch.tensor(X_cat_train, dtype=torch.long), 
                             torch.tensor(y_train_enc, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    test_ds = TensorDataset(torch.tensor(X_num_test, dtype=torch.float32), 
                            torch.tensor(X_cat_test, dtype=torch.long), 
                            torch.tensor(y_test_enc, dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    num_cols, cat_cols = X_num_train.shape[1], X_cat_train.shape[1]

    # --- 3. Inicialización de Modelos y Optimizador ---
    if architecture == 'mlp':
        # Instancia el modelo MLP-VAE (sin Transformers).
        model = MLP_ModelVAE(
            d_numerical=num_cols, categories=categories, d_token=d_token,
            n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim,
            mlp_dropout=mlp_dropout, bias=token_bias, num_classes=None
        ).to(device)
        pre_encoder = MLP_EncoderModel(
            d_numerical=num_cols, categories=categories, d_token=d_token,
            n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim,
            mlp_dropout=mlp_dropout
        ).to(device).eval()
        pre_decoder = MLP_DecoderModel(
            d_numerical=num_cols, categories=categories, d_token=d_token,
            n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim,
            mlp_dropout=mlp_dropout
        ).to(device).eval()
    else:
        # Instancia el modelo VAE con Transformer (comportamiento original).
        model = ModelVAE(num_layers, num_cols, categories, d_token, num_classes=None, n_head=n_head, factor=factor, bias=token_bias).to(device)
        pre_encoder = EncoderModel(num_layers, num_cols, categories, d_token, n_head=n_head, factor=factor).to(device).eval()
        pre_decoder = DecoderModel(num_layers, num_cols, categories, d_token, n_head=n_head, factor=factor).to(device).eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_pretrain, weight_decay=wd_pretrain)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)
    tracker = LossTracker(log_dir=os.path.join(ckpt_dir, 'runs', f"seed_{random_state}"))
    
    # --- 4. Fase 1: Pre-entrenamiento del VAE (sin supervisión) ---
    print("\n--- Iniciando Fase 1: Pre-entrenamiento No Supervisado ---")
    beta = max_beta
    best_val_loss = float('inf')
    early_stop_counter = 0
    start_time = time.time()
    
    for epoch in range(1, pretrain_epochs + 1):
        # Ejecuta un ciclo de entrenamiento y validación.
        train_mse, train_ce, train_kld, train_loss, _ = train_vae(model, train_loader, optimizer, device, beta)
        val_mse, val_ce, val_kld, val_loss, _ = validate_vae(model, test_loader, device, beta)
        tracker.log(train_mse, train_ce, train_kld, 0, train_loss, val_mse, val_ce, val_kld, 0, val_loss)
        
        # Actualiza el scheduler con la pérdida de validación.
        scheduler.step(val_loss)

        # Lógica para guardar el mejor modelo basado en la pérdida de validación.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            early_stop_counter += 1

        # Implementa el early stopping para evitar sobreajuste.
        if early_stop_counter >= early_stop_counter_pretrain:
            print(f"No hubo mejora en {early_stop_counter_pretrain} épocas. Early stopping en epoch {epoch}.")
            break

        print(f"Epoch {epoch:03d}/{pretrain_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    pretrain_time = time.time() - start_time
    print(f"--- Fase 1 Finalizada. Tiempo: {pretrain_time:.2f} segundos ---")


    # --- 5. Fase 2: Fine-Tuning Supervisado (Opcional) ---
    finetune_time = 0
    if finetune_epochs > 0:
        print("\n--- Iniciando Fase 2: Fine-Tuning Supervisado ---")

        # Carga el mejor modelo pre-entrenado y le añade el cabezal de clasificación.
        if architecture == 'mlp':
            ft_model = MLP_ModelVAE(
                d_numerical=num_cols, categories=categories, d_token=d_token,
                n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim,
                mlp_dropout=mlp_dropout, bias=token_bias,
                num_classes=num_classes, dropout_class=dropout_ft
            ).to(device)
        else:
            ft_model = ModelVAE(
                num_layers=num_layers, d_numerical=num_cols, categories=categories, d_token=d_token,
                num_classes=num_classes, n_head=n_head, factor=factor, bias=token_bias, droput_class=dropout_ft
            ).to(device)
        ft_model.load_state_dict(torch.load(model_save_path), strict=False)

        # Define un nuevo optimizador para el fine-tuning, usualmente con un learning rate más bajo.
        optimizer_ft = torch.optim.AdamW(ft_model.parameters(), lr=lr_ft, weight_decay=wd_ft)
        scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.9, patience=5, verbose=True)

        beta = min_beta
        best_val_loss_ft = float('inf')
        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(1, finetune_epochs + 1):
            train_mse, train_ce, train_kld, train_loss, train_class_loss = train_vae(ft_model, train_loader, optimizer_ft, device, beta, alpha=alpha_ft)
            val_mse, val_ce, val_kld, val_loss, val_class_loss = validate_vae(ft_model, test_loader, device, beta, alpha=alpha_ft)
            tracker.log(train_mse, train_ce, train_kld, train_class_loss, train_loss, val_mse, val_ce, val_kld, val_class_loss, val_loss)
            scheduler_ft.step(val_loss)

            if val_loss < best_val_loss_ft:
                best_val_loss_ft = val_loss
                torch.save(ft_model.state_dict(), ft_model_save_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_counter_ft:
                print(f"No hubo mejora en {early_stop_counter_ft} épocas. Early stopping en epoch {epoch}.")
                break
            
            print(f"FT Epoch {epoch:03d}/{finetune_epochs} | Val Loss: {val_loss:.4f} (Recon: {val_mse+val_ce:.4f}, Class: {val_class_loss:.4f})")
        
        finetune_time = time.time() - start_time
        print(f"--- Fase 2 Finalizada. Tiempo: {finetune_time:.2f} segundos ---")
        model_save_path = ft_model_save_path
        model = ft_model
    else:
        print("\n--- Omitiendo Fase 2: Fine-Tuning Supervisado ---")

    # --- 6. Guardado Final de Artefactos ---
    # Guarda los modelos Encoder y Decoder por separado y las representaciones latentes.
    train_z, encoder_inference_time = save_best_encoder_decoder(
        model=model, model_save_path=model_save_path, num_cols=num_cols, categories=categories,
        pre_encoder=pre_encoder, pre_decoder=pre_decoder, num_scaler=num_scaler, cat_encoder=cat_encoder, 
        label_encoder=label_encoder, 
        X_train_num=torch.tensor(X_num_train, dtype=torch.float32).to(device), 
        X_train_cat=torch.tensor(X_cat_train, dtype=torch.long).to(device),
        y_train_enc=y_train_enc, 
        X_test_num=torch.tensor(X_num_test, dtype=torch.float32).to(device), 
        X_test_cat=torch.tensor(X_cat_test, dtype=torch.long).to(device),
        y_test_enc=y_test_enc, 
        ckpt_dir=ckpt_dir, device=device, random_state=random_state
    )

    return train_z, y_train_enc, pretrain_time, finetune_time, encoder_inference_time
