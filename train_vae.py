import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import preprocessing, read_json_config
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
from models import LossTracker

warnings.filterwarnings('ignore')
import time

def train_vae(model, loader, optimizer, device, beta=1.0, alpha=0.0):
    """Entrena una epoch de un VAE y devuelve las pérdidas medias: mse, ce, kld y total."""
    model.train()
    total_mse = total_ce = total_kld = total_loss = total_class_loss = 0.0
    n_samples = 0
    loader = tqdm(loader, total=len(loader))
    for X_num_batch, X_cat_batch, y_batch in loader:
        batch_size = X_num_batch.size(0)
        n_samples += batch_size

        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass del modelo
        recon_num, recon_cat, mu_z, logvar_z, class_logits = model(X_num_batch, X_cat_batch)

        # Cálculo de pérdidas parciales
        loss_mse, loss_ce, loss_kld, _ = compute_loss(
            X_num_batch, X_cat_batch,
            recon_num, recon_cat,
            mu_z, logvar_z
        )

        # Pérdida total con KL ponderado
        loss = loss_mse + loss_ce + beta * loss_kld

        loss_class = torch.tensor(0.0).to(device) 
        # Cálculo de la pérdida de clasificación si alpha > 0 y hay clasificador
        if alpha > 0 and class_logits is not None:
            # Asegúrate que y_batch tenga el tipo correcto (LongTensor para CrossEntropy)
            y_batch = y_batch.long()
            loss_class = F.cross_entropy(class_logits, y_batch)
            loss = loss + alpha * loss_class
        elif alpha > 0 and class_logits is None:
             print("Advertencia: alpha > 0 pero el modelo no tiene cabeza clasificadora o no devolvió logits.")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumulación de pérdidas totales
        total_mse  += loss_mse.item() * batch_size
        total_ce   += loss_ce.item()  * batch_size
        total_kld  += loss_kld.item() * batch_size
        total_class_loss += loss_class.item() * batch_size
        total_loss += loss.item()     * batch_size

    # Cálculo de promedios por muestra
    avg_mse   = total_mse  / n_samples
    avg_ce    = total_ce   / n_samples
    avg_kld   = total_kld  / n_samples
    avg_total = total_loss / n_samples
    avg_class_loss = total_class_loss / n_samples

    return avg_mse, avg_ce, avg_kld, avg_total, avg_class_loss

def validate_vae(model, loader, device, beta=1.0, alpha=0.0):
    """Evalúa el modelo VAE y devuelve las pérdidas medias: mse, ce, kld y total."""
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

            # Forward pass sin gradientes
            recon_num, recon_cat, mu_z, logvar_z, class_logits = model(X_num_batch, X_cat_batch)
            #print("Recon num: ", recon_num)
            #print("Recon cat: ", recon_cat)
            # Cálculo de pérdidas parciales
            loss_mse, loss_ce, loss_kld, _ = compute_loss(
                X_num_batch, X_cat_batch,
                recon_num, recon_cat,
                mu_z, logvar_z
            )
            
            # Pérdida total con KL ponderado
            loss = loss_mse + loss_ce + beta * loss_kld

            loss_class = torch.tensor(0.0).to(device) 
            # Cálculo de la pérdida de clasificación si alpha > 0 y hay clasificador
            if alpha > 0 and class_logits is not None:
                # Asegúrate que y_batch tenga el tipo correcto (LongTensor para CrossEntropy)
                y_batch = y_batch.long()
                loss_class = F.cross_entropy(class_logits, y_batch)
                loss = loss + alpha * loss_class
            elif alpha > 0 and class_logits is None:
                print("Advertencia: alpha > 0 pero el modelo no tiene cabeza clasificadora o no devolvió logits.")


            # Acumulación de pérdidas totales
            total_mse  += loss_mse.item() * batch_size
            total_ce   += loss_ce.item()  * batch_size
            total_kld  += loss_kld.item() * batch_size
            total_class_loss += loss_class.item() * batch_size
            total_loss += loss.item()     * batch_size
            

    # Cálculo de promedios por muestra
    avg_mse   = total_mse  / n_samples
    avg_ce    = total_ce   / n_samples
    avg_kld   = total_kld  / n_samples
    avg_total = total_loss / n_samples
    avg_class_loss = total_class_loss / n_samples
    #print("avg_mse: ", avg_mse)
    return avg_mse, avg_ce, avg_kld, avg_total, avg_class_loss

def save_best_encoder_decoder(model, model_save_path, num_cols, categories, pre_encoder, pre_decoder, num_scaler, cat_encoder, label_encoder,
                               X_train_num, X_train_cat, y_train_enc, X_test_num, X_test_cat, y_test_enc, ckpt_dir, device, random_state=None):
    """
    Recarga el mejor modelo guardado, extrae pesos para pre_encoder y pre_decoder,
    guarda sus pesos, y guarda las embeddings latentes (train_z.npy).
    
    Guarda encoder.pt y decoder.pt dentro de ckpt_dir.
    """

    # Definir paths de guardado
    encoder_save_path = os.path.join(ckpt_dir, f'encoder_seed_{random_state}.pt')
    decoder_save_path = os.path.join(ckpt_dir, f'decoder_seed_{random_state}.pt')

    # 1. Recargar el mejor modelo
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    with open(os.path.join(ckpt_dir, f'pre_encoders_seed_{random_state}.pkl'), 'wb') as f:
        pickle.dump({
            "num_cols": num_cols,
            "categories": categories,
            'num_scaler': num_scaler,
            'cat_ordinal_encoder': cat_encoder,
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
        start = time.time()
        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
        end = time.time()
        encoder_inference_time = (end - start) / X_train_num.shape[0]

        test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()

        # 6. Guardar las representaciones latentes
        if random_state == None:
            np.save(os.path.join(ckpt_dir, f'train_z.npy'), train_z)
            np.save(os.path.join(ckpt_dir, f'train_y.npy'), y_train_enc)
        else:
            np.save(os.path.join(ckpt_dir, f'train_z_seed_{random_state}.npy'), train_z)
            np.save(os.path.join(ckpt_dir, f'train_y_seed_{random_state}.npy'), y_train_enc)
            np.save(os.path.join(ckpt_dir, f'test_z_seed_{random_state}.npy'), test_z)
            np.save(os.path.join(ckpt_dir, f'test_y_seed_{random_state}.npy'), y_test_enc)

        print('Successfully saved best encoder, decoder, and latent embeddings!')

        return train_z, encoder_inference_time

def main(config, base_dir, set_data, encoding='ordinal', random_state=0, batch_size=64, pretrain_epochs=50, finetune_epochs=30, concat_label=False,ckpt_dir='ckpt', hyperparams = {}):
    
    # Hiperparametros de VAE
    max_beta = hyperparams.get('max_beta',1e-2)
    min_beta = hyperparams.get('min_beta',1e-5)
    lambda_ = hyperparams.get('lambda_',0.7)
    lr_pretrain = hyperparams.get('lr_pretrain',1e-3)
    wd_pretrain =hyperparams.get('wd_pretrain',0)
    d_token =hyperparams.get('d_token',4)
    token_bias =hyperparams.get('token_bias',True)
    n_head = hyperparams.get('n_head',1)
    factor = hyperparams.get('factor',32)
    num_layers =hyperparams.get('num_layers',2)
    early_stop_counter_pretrain = hyperparams.get('early_stop_counter_pretrain',20)

    # Hiperparametros de la cabeza clasificadora
    dropout_ft = hyperparams.get('dropout_ft',0.3)
    alpha_ft = hyperparams.get("alpha_ft",1e-5)
    lr_ft = hyperparams.get("lr_ft",1e-4)
    wd_ft = hyperparams.get('wd_ft',0)
    early_stop_counter_ft = hyperparams.get("early_stop_counter_ft",20)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_save_path = os.path.join(ckpt_dir, f'best_vae_seed_{random_state}.pt')
    ft_model_save_path = os.path.join(ckpt_dir, f'ft_best_vae_seed_{random_state}.pt')

    X_train, y_train, X_test, y_test = set_data
    (
        X_num_train, X_cat_train,
        X_num_test,  X_cat_test,
        y_train_enc, y_test_enc,
        num_scaler,  cat_encoder,
        label_encoder
                        ) = preprocessing(config, X_train, y_train, X_test, y_test, encoding=encoding, concat_label=concat_label, random_state=random_state)

    categories = [len(set(X_cat_train[:, i])) for i in range(X_cat_train.shape[1])]
    num_classes = len(set(y_train_enc))

    # Tensores de entrenamiento
    X_num_train = torch.tensor(X_num_train, dtype=torch.float32)
    X_cat_train = torch.tensor(X_cat_train, dtype=torch.long)
    y_train_enc = torch.tensor(y_train_enc, dtype=torch.long)
    train_ds = TensorDataset(X_num_train, X_cat_train, y_train_enc)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # Tensores de validación/test
    X_num_test = torch.tensor(X_num_test, dtype=torch.float32)
    X_cat_test = torch.tensor(X_cat_test, dtype=torch.long)
    y_test_enc = torch.tensor(y_test_enc, dtype=torch.long)
    test_ds = TensorDataset(X_num_test, X_cat_test, y_test_enc)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    num_cols, cat_cols = X_num_train.shape[1], X_cat_train.shape[1]   

    model = ModelVAE(num_layers, num_cols, categories, d_token, num_classes=None, n_head = n_head, factor = factor, bias = True)
    model = model.to(device)
    print(model)
    pre_encoder = EncoderModel(num_layers, num_cols, categories, d_token, n_head = n_head, factor = factor).to(device)
    pre_decoder = DecoderModel(num_layers, num_cols, categories, d_token, n_head = n_head, factor = factor).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_pretrain, weight_decay=wd_pretrain)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    tracker = LossTracker(log_dir=os.path.join(ckpt_dir, 'runs',f"seed_{random_state}"))

    beta = max_beta
    patience = 0

    best_val_loss = float('inf')
    early_stop_counter = 0

    # =======================================
    # --- STAGE 1: Pretrain ---
    # =======================================

    # 5. Bucle de entrenamiento
    start = time.time()
    for epoch in range(1, pretrain_epochs+1):
        # Training
        train_mse, train_ce, train_kld, train_loss, _ = \
            train_vae(model, train_loader, optimizer, device, beta)

        # Validación
        val_mse, val_ce, val_kld, val_loss, _ = \
            validate_vae(model, test_loader, device, beta)

        tracker.log(train_mse, train_ce, train_kld, 0, train_loss,
                            val_mse,   val_ce,   val_kld,   0,   val_loss)

        # Ajuste de learning rate
        scheduler.step(val_loss)

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            patience = 0
            torch.save(model.state_dict(),model_save_path)

        else:
            early_stop_counter += 1
            patience += 1
            if patience == 10:
                if beta > min_beta:
                    beta = beta * lambda_

        if early_stop_counter >= early_stop_counter_pretrain:
            print(f"No hubo mejora en {early_stop_counter_pretrain} épocas. Early stopping en epoch {epoch}.")
            break

        # Impresión de métricas
        print(
            f"Epoch {epoch:03d}/{pretrain_epochs} "
            f"Train Loss: {train_loss:.6f} (MSE:{train_mse:.6f}, CE:{train_ce:.6f}, KLD:{train_kld:.6f})  "
            f"Val Loss: {val_loss:.6f} (MSE:{val_mse:.6f}, CE:{val_ce:.6f}, KLD:{val_kld:.6f})"
        )
    end = time.time()
    pretrain_time = end - start
    print(f"Training time: {end - start:.2f} seconds")


    # =======================================
    # --- STAGE 2: Supervised Fine-tuning ---
    # =======================================
    finetune_time = 0
    if finetune_epochs > 0:
        print("\n--- Starting Stage 2: Supervised Fine-tuning ---")

        # Crear modelo CON cabeza clasificadora
        ft_model = ModelVAE(
            num_layers=num_layers, d_numerical=num_cols, categories=categories, d_token=d_token,
            num_classes=num_classes, # AHORA con cabeza clasificadora
            n_head=n_head, factor=factor, bias=token_bias, droput_class=dropout_ft
        )
        ft_model = ft_model.to(device)
        ft_model.load_state_dict(torch.load(model_save_path), strict=False)

        print("Fine-tuning model architecture:")
        #print(ft_model)

        # Nuevo optimizador para fine-tuning (todos los parámetros, LR más bajo)
        optimizer_ft = torch.optim.AdamW(ft_model.parameters(), lr=lr_ft, weight_decay=wd_ft)
        scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.9, patience=5, verbose=True) # Menor paciencia para FT

        beta = min_beta # Empezar con beta bajo o mantener el final del pre-entrenamiento? Aquí usamos MIN_BETA
        patience = 0
        best_val_loss_ft = float('inf')
        early_stop_counter = 0

        start = time.time()
        for epoch in range(1, finetune_epochs + 1):
            # Entrenar con alpha > 0
            train_mse, train_ce, train_kld, train_loss, train_class_loss = \
                train_vae(ft_model, train_loader, optimizer_ft, device, beta, alpha=alpha_ft)

            # Validar con alpha > 0
            val_mse, val_ce, val_kld, val_loss, val_class_loss = \
                validate_vae(ft_model, test_loader, device, beta, alpha=alpha_ft)

            tracker.log(train_mse, train_ce, train_kld, train_class_loss, train_loss,
                                val_mse,   val_ce,   val_kld,   val_class_loss,   val_loss)

            scheduler_ft.step(val_loss) # Usar scheduler de fine-tuning

            if val_loss < best_val_loss_ft:
                print(f"Stage 2 - Epoch {epoch}: Val loss improved to {val_loss:.6f}. Saving fine-tuned model...")
                best_val_loss_ft = val_loss
                torch.save(ft_model.state_dict(), ft_model_save_path)
                early_stop_counter = 0
                patience = 0

            else:
                early_stop_counter += 1
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambda_

            if early_stop_counter >= early_stop_counter_ft:
                print(f"No hubo mejora en {early_stop_counter_ft} épocas. Early stopping en epoch {epoch}.")
                break

            print(f"Stage 2 - Epoch {epoch:03d}/{finetune_epochs} | Beta: {beta:.2e} | Alpha: {alpha_ft:.2f} | Loss: {train_loss:.4f} (MSE:{train_mse:.4f}, CE:{train_ce:.4f}, KLD:{train_kld:.4f}, ClassL:{train_class_loss:.4f}) | Val Loss: {val_loss:.4f} (MSE:{val_mse:.4f}, CE:{val_ce:.4f}, KLD:{val_kld:.4f}, ClassL:{val_class_loss:.4f})")

        print(f"--- Stage 2 Finished. Best fine-tuning validation loss: {best_val_loss_ft:.6f} ---")
        end = time.time()
        finetune_time = end - start
        model_save_path = ft_model_save_path
        model = ft_model
    else:
        print("\n--- Skipping Stage 2: Supervised Fine-tuning (perform_fine_tune=False) ---")

    # Guardando información de encoder decoder
    train_z, encoder_inference_time = save_best_encoder_decoder(
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
        X_test_num=X_num_test.to(device),
        X_test_cat=X_cat_test.to(device),
        y_test_enc=y_test_enc,
        ckpt_dir=ckpt_dir,
        device=device,
        random_state=random_state
    )

    return train_z, y_train_enc, pretrain_time, finetune_time, encoder_inference_time
    
if __name__ == "__main__":
    dataset_path = '/mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/metadata.json'
    encoding = 'ordinal'
    random_state = 0
    ckpt_dir = 'ckpt_custom_model'
    main(dataset_path, encoding, random_state, ckpt_dir=ckpt_dir)