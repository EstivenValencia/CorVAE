import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import preprocessing
from models import ModelVAE, compute_loss, EncoderModel, DecoderModel

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

import torch.nn.functional as F
import pickle

warnings.filterwarnings('ignore')

MAX_BETA = 1e-2
MIN_BETA = 1e-5
LAMBDA = 0.7

LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2

# Parametros de entrenamiento
num_epochs = 4000
batch_size = 64

def train_vae(model, loader, optimizer, device, beta=1.0):
    """Entrena una epoch de un VAE y devuelve las pérdidas medias: mse, ce, kld y total."""
    model.train()
    total_mse = total_ce = total_kld = total_loss = 0.0
    n_samples = 0
    loader = tqdm(loader, total=len(loader))
    for X_num_batch, X_cat_batch in loader:
        batch_size = X_num_batch.size(0)
        n_samples += batch_size

        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)

        # Forward pass del modelo
        recon_num, recon_cat, mu_z, logvar_z = model(X_num_batch, X_cat_batch)

        # Cálculo de pérdidas parciales
        loss_mse, loss_ce, loss_kld, _ = compute_loss(
            X_num_batch, X_cat_batch,
            recon_num, recon_cat,
            mu_z, logvar_z
        )

        # Pérdida total con KL ponderado
        loss = loss_mse + loss_ce + beta * loss_kld

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumulación de pérdidas totales
        total_mse  += loss_mse.item() * batch_size
        total_ce   += loss_ce.item()  * batch_size
        total_kld  += loss_kld.item() * batch_size
        total_loss += loss.item()     * batch_size

    # Cálculo de promedios por muestra
    avg_mse   = total_mse  / n_samples
    avg_ce    = total_ce   / n_samples
    avg_kld   = total_kld  / n_samples
    avg_total = total_loss / n_samples

    return avg_mse, avg_ce, avg_kld, avg_total

def validate_vae(model, loader, device, beta=1.0):
    """Evalúa el modelo VAE y devuelve las pérdidas medias: mse, ce, kld y total."""
    model.eval()
    total_mse = total_ce = total_kld = total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_num_batch, X_cat_batch in loader:
            batch_size = X_num_batch.size(0)
            n_samples += batch_size

            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)

            # Forward pass sin gradientes
            recon_num, recon_cat, mu_z, logvar_z = model(X_num_batch, X_cat_batch)

            # Cálculo de pérdidas parciales
            loss_mse, loss_ce, loss_kld, _ = compute_loss(
                X_num_batch, X_cat_batch,
                recon_num, recon_cat,
                mu_z, logvar_z
            )

            # Pérdida total con KL ponderado
            loss = loss_mse + loss_ce + beta * loss_kld

            # Acumulación de pérdidas totales
            total_mse  += loss_mse.item() * batch_size
            total_ce   += loss_ce.item()  * batch_size
            total_kld  += loss_kld.item() * batch_size
            total_loss += loss.item()     * batch_size

    # Cálculo de promedios por muestra
    avg_mse   = total_mse  / n_samples
    avg_ce    = total_ce   / n_samples
    avg_kld   = total_kld  / n_samples
    avg_total = total_loss / n_samples

    return avg_mse, avg_ce, avg_kld, avg_total

def save_best_encoder_decoder(model, model_save_path, num_cols, categories, pre_encoder, pre_decoder, num_scaler, cat_encoder, label_encoder,
                               X_train_num, X_train_cat, y_train_enc, ckpt_dir, device):
    """
    Recarga el mejor modelo guardado, extrae pesos para pre_encoder y pre_decoder,
    guarda sus pesos, y guarda las embeddings latentes (train_z.npy).
    
    Guarda encoder.pt y decoder.pt dentro de ckpt_dir.
    """

    # Definir paths de guardado
    encoder_save_path = os.path.join(ckpt_dir, 'encoder.pt')
    decoder_save_path = os.path.join(ckpt_dir, 'decoder.pt')

    # 1. Recargar el mejor modelo
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    with open(os.path.join(ckpt_dir, 'pre_encoders.pkl'), 'wb') as f:
        pickle.dump({
            "num_cols": num_cols,
            "categories": categories,
            'num_scaler': num_scaler,
            'cat_encoder': cat_encoder,
            'label_encoder': label_encoder
        }, f)

    with torch.no_grad():
        # 2. Cargar pesos del modelo al pre_encoder y pre_decoder
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        # 3. Guardar los pesos del pre_encoder y pre_decoder
        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        # 5. Obtener representaciones latentes
        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        # 6. Guardar las representaciones latentes
        np.save(os.path.join(ckpt_dir, 'train_z.npy'), train_z)
        np.save(os.path.join(ckpt_dir, 'train_y.npy'), y_train_enc)

        print('Successfully saved best encoder, decoder, and latent embeddings!')

def main(dataset_path, encoding='ordinal', random_state=0, batch_size=64, ckpt_dir='ckpt', fine_tune= False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(ckpt_dir, exist_ok=True)
    model_save_path = os.path.join(ckpt_dir, 'best_vae.pt')

    (
        X_num_train, X_cat_train,
        X_num_test,  X_cat_test,
        y_train_enc, y_test_enc,
        num_scaler,  cat_encoder,
        label_encoder
                        ) = preprocessing(dataset_path, encoding='ordinal', random_state=random_state)

    # Tensores de entrenamiento
    X_num_train = torch.tensor(X_num_train, dtype=torch.float32)
    X_cat_train = torch.tensor(X_cat_train, dtype=torch.long)
    train_ds = TensorDataset(X_num_train, X_cat_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Tensores de validación/test
    X_num_test = torch.tensor(X_num_test, dtype=torch.float32)
    X_cat_test = torch.tensor(X_cat_test, dtype=torch.long)
    test_ds = TensorDataset(X_num_test, X_cat_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_cols, cat_cols = X_num_train.shape[1], X_cat_train.shape[1]    
    categories = [len(set(X_cat_train[:, i])) for i in range(X_cat_train.shape[1])]

    model = ModelVAE(NUM_LAYERS, num_cols, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True)
    model = model.to(device)

    pre_encoder = EncoderModel(NUM_LAYERS, num_cols, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_decoder = DecoderModel(NUM_LAYERS, num_cols, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    beta = MAX_BETA
    patience = 0

    best_val_loss = float('inf')
    # 5. Bucle de entrenamiento
    start = time.time()
    for epoch in range(1, num_epochs+1):
        # Training
        train_mse, train_ce, train_kld, train_loss = \
            train_vae(model, train_loader, optimizer, device, beta)

        # Validación
        val_mse, val_ce, val_kld, val_loss = \
            validate_vae(model, test_loader, device, beta)

        # Ajuste de learning rate
        scheduler.step(val_loss)

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(),model_save_path)
        else:
            patience += 1
            if patience == 10:
                if beta > MIN_BETA:
                    beta = beta * LAMBDA

        # Impresión de métricas
        print(
            f"Epoch {epoch:03d}/{num_epochs} "
            f"Train Loss: {train_loss:.6f} (MSE:{train_mse:.6f}, CE:{train_ce:.6f}, KLD:{train_kld:.6f})  "
            f"Val Loss: {val_loss:.6f} (MSE:{val_mse:.6f}, CE:{val_ce:.6f}, KLD:{val_kld:.6f})"
        )
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")

    save_best_encoder_decoder(
        model=model,
        model_save_path=model_save_path,
        num_cols=num_cols, 
        categories=categories,
        pre_encoder=pre_encoder,
        pre_decoder=pre_decoder,
        num_scaler=num_scaler, 
        cat_encoder=cat_encoder, 
        label_encoder=label_encoder,
        X_train_num=X_num_train.to(device),
        X_train_cat=X_cat_train.to(device),
        y_train_enc=y_train_enc,
        ckpt_dir=ckpt_dir,
        device=device
    )
    
if __name__ == "__main__":
    dataset_path = '/mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/metadata.json'
    encoding = 'ordinal'
    random_state = 0
    ckpt_dir = 'ckpt_custom_model'
    main(dataset_path, encoding, random_state, ckpt_dir=ckpt_dir)