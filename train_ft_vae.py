import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import preprocessing # Asegúrate que esto exista y funcione
# Asume que tus modelos están definidos como en la respuesta anterior
# from models import ModelVAE, compute_loss, EncoderModel, DecoderModel, ClassifierHead
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm
import json
import time
import warnings
import torch.nn.functional as F # Asegúrate que F está importado

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


warnings.filterwarnings('ignore')

# --- Constantes (Ajusta según necesidad) ---
MAX_BETA = 1e-2
MIN_BETA = 1e-5
LAMBDA_BETA = 0.7 # Renombrado para evitar confusión con lambda de python
LR_PRETRAIN = 1e-3
LR_FINETUNE = 1e-4 # Learning rate más bajo para fine-tuning
WD = 0
D_TOKEN = 64       # Aumentado como ejemplo, ajusta según tus necesidades
TOKEN_BIAS = True
N_HEAD = 2         # Aumentado como ejemplo
FACTOR = 4         # Ajustado como ejemplo
NUM_LAYERS = 2
ALPHA_FT = 0.5     # Peso para la pérdida de clasificación en fine-tuning
PATIENCE_PRETRAIN = 15
PATIENCE_FINETUNE = 10



# Define el peso para la pérdida de clasificación (ajusta según sea necesario)
ALPHA_FT = 0.7 # Valor mencionado en el paper [cite: 97, 375]

def train_vae(model, loader, optimizer, device, beta=1.0, alpha=0.0):
    """
    Entrena una epoch de un VAE, opcionalmente con fine-tuning supervisado.

    Args:
        model (nn.Module): El modelo ModelVAE (que puede incluir ClassifierHead).
        loader (DataLoader): DataLoader para los datos de entrenamiento (debe incluir X_num, X_cat, y).
        optimizer (optim.Optimizer): Optimizador.
        device (torch.device): Dispositivo ('cuda' o 'cpu').
        beta (float): Peso para la pérdida KLD.
        alpha (float): Peso para la pérdida de clasificación (Cross Entropy). Si es 0, se ignora.

    Returns:
        tuple: Pérdidas medias: mse, ce, kld, class_loss (si alpha > 0), total.
    """
    model.train()
    total_mse = total_ce = total_kld = total_class_loss = total_loss = 0.0
    n_samples = 0
    loader_tqdm = tqdm(loader, total=len(loader), desc="Training")

    for batch in loader_tqdm:
        X_num_batch, X_cat_batch, y_batch = batch # Ahora el loader devuelve 3 elementos
        batch_size_current = X_num_batch.size(0)
        n_samples += batch_size_current

        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)
        y_batch = y_batch.to(device) # Mover etiquetas al dispositivo

        # Forward pass del modelo
        recon_num, recon_cat, mu_z, logvar_z, class_logits = model(X_num_batch, X_cat_batch)

        # Cálculo de pérdidas parciales del VAE (MSE, CE reconstrucción, KLD)
        # Asegúrate que tu `compute_loss` devuelve (mse, ce, kld, acc)
        loss_mse, loss_ce, loss_kld, _ = compute_loss(
            X_num_batch, X_cat_batch,
            recon_num, recon_cat,
            mu_z, logvar_z # Pasamos mu_z y logvar_z completos
        )

        # Pérdida base del VAE
        vae_loss = loss_mse + loss_ce + beta * loss_kld
        loss = vae_loss
        loss_class = torch.tensor(0.0).to(device) # Inicializar

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
        # Optional: Gradient clipping si es necesario
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Acumulación de pérdidas totales
        total_mse += loss_mse.item() * batch_size_current
        total_ce += loss_ce.item() * batch_size_current
        total_kld += loss_kld.item() * batch_size_current
        total_class_loss += loss_class.item() * batch_size_current
        total_loss += loss.item() * batch_size_current

        # Actualizar barra de progreso
        log_dict = {'Loss': loss.item(), 'MSE': loss_mse.item(), 'CE': loss_ce.item(), 'KLD': loss_kld.item()}
        if alpha > 0:
            log_dict['ClassLoss'] = loss_class.item()
        loader_tqdm.set_postfix(log_dict)

    # Cálculo de promedios por muestra
    avg_mse = total_mse / n_samples
    avg_ce = total_ce / n_samples
    avg_kld = total_kld / n_samples
    avg_class_loss = total_class_loss / n_samples
    avg_total = total_loss / n_samples

    return avg_mse, avg_ce, avg_kld, avg_class_loss, avg_total

def validate_vae(model, loader, device, beta=1.0, alpha=0.0):
    """
    Evalúa el modelo VAE, opcionalmente con fine-tuning supervisado.

    Args:
        model (nn.Module): El modelo ModelVAE.
        loader (DataLoader): DataLoader para los datos de validación (debe incluir X_num, X_cat, y).
        device (torch.device): Dispositivo ('cuda' o 'cpu').
        beta (float): Peso para la pérdida KLD.
        alpha (float): Peso para la pérdida de clasificación. Si es 0, se ignora.

    Returns:
        tuple: Pérdidas medias: mse, ce, kld, class_loss (si alpha > 0), total.
    """
    model.eval()
    total_mse = total_ce = total_kld = total_class_loss = total_loss = 0.0
    n_samples = 0
    loader_tqdm = tqdm(loader, total=len(loader), desc="Validation")

    with torch.no_grad():
        for batch in loader_tqdm:
            X_num_batch, X_cat_batch, y_batch = batch
            batch_size_current = X_num_batch.size(0)
            n_samples += batch_size_current

            X_num_batch = X_num_batch.to(device)
            X_cat_batch = X_cat_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass sin gradientes
            recon_num, recon_cat, mu_z, logvar_z, class_logits = model(X_num_batch, X_cat_batch)

            # Cálculo de pérdidas parciales del VAE
            loss_mse, loss_ce, loss_kld, _ = compute_loss(
                X_num_batch, X_cat_batch,
                recon_num, recon_cat,
                mu_z, logvar_z
            )

            # Pérdida base del VAE
            vae_loss = loss_mse + loss_ce + beta * loss_kld
            loss = vae_loss
            loss_class = torch.tensor(0.0).to(device)

            # Cálculo de la pérdida de clasificación si alpha > 0
            if alpha > 0 and class_logits is not None:
                y_batch = y_batch.long()
                loss_class = F.cross_entropy(class_logits, y_batch)
                loss = loss + alpha * loss_class
            elif alpha > 0 and class_logits is None:
                 print("Advertencia: alpha > 0 pero el modelo no tiene cabeza clasificadora o no devolvió logits.")


            # Acumulación de pérdidas totales
            total_mse += loss_mse.item() * batch_size_current
            total_ce += loss_ce.item() * batch_size_current
            total_kld += loss_kld.item() * batch_size_current
            total_class_loss += loss_class.item() * batch_size_current
            total_loss += loss.item() * batch_size_current

            # Actualizar barra de progreso
            log_dict = {'Loss': loss.item(), 'MSE': loss_mse.item(), 'CE': loss_ce.item(), 'KLD': loss_kld.item()}
            if alpha > 0:
                log_dict['ClassLoss'] = loss_class.item()
            loader_tqdm.set_postfix(log_dict)

    # Cálculo de promedios por muestra
    avg_mse = total_mse / n_samples
    avg_ce = total_ce / n_samples
    avg_kld = total_kld / n_samples
    avg_class_loss = total_class_loss / n_samples
    avg_total = total_loss / n_samples

    return avg_mse, avg_ce, avg_kld, avg_class_loss, avg_total


def load_pretrained_weights(model_to_load, pretrained_path, device):
    """
    Carga pesos pre-entrenados en un modelo, ignorando claves que no coincidan
    (útil para cargar pesos VAE en un modelo VAE+Clasificador).
    """
    try:
        print(f"Attempting to load pretrained weights from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model_to_load.state_dict()

        # 1. Filtrar claves no necesarias (ej. del clasificador si no están en el pre-entrenado)
        #    y claves que no estén en el modelo actual (menos probable)
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        # 2. Actualizar el diccionario del modelo actual con los pesos cargados
        model_dict.update(pretrained_dict_filtered)
        # 3. Cargar el state_dict actualizado
        model_to_load.load_state_dict(model_dict)
        print(f"Successfully loaded {len(pretrained_dict_filtered)} parameter tensors from {pretrained_path}")
        # Informar sobre claves ignoradas
        ignored_keys = [k for k in pretrained_dict if k not in pretrained_dict_filtered]
        if ignored_keys:
            print(f"Ignored keys (not found or size mismatch): {ignored_keys}")
        missing_keys = [k for k in model_dict if k not in pretrained_dict_filtered]
        if missing_keys:
            print(f"Missing keys in loaded dict (likely classifier): {missing_keys}")

    except FileNotFoundError:
        print(f"Error: Pretrained weights file not found at {pretrained_path}. Skipping weight loading.")
    except Exception as e:
        print(f"Error loading pretrained weights: {e}. Skipping weight loading.")


def main(dataset_path, encoding='ordinal', random_state=0, batch_size=64,
         pretrain_epochs=50, finetune_epochs=30, # Épocas separadas
         ckpt_dir='ckpt', perform_fine_tune=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Paths para los modelos ---
    os.makedirs(ckpt_dir, exist_ok=True)
    pretrained_vae_path = os.path.join(ckpt_dir, 'best_vae_pretrained.pt')
    finetuned_model_path = os.path.join(ckpt_dir, 'best_model_finetuned.pt')
    final_model_path = finetuned_model_path if perform_fine_tune else pretrained_vae_path

    # --- Carga y Preprocesamiento de Datos ---
    (
        X_num_train, X_cat_train,
        X_num_test,  X_cat_test,
        y_train_enc, y_test_enc,
        num_scaler,  cat_encoder,
        label_encoder
    ) = preprocessing(dataset_path, encoding=encoding, random_state=random_state)

    y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    X_num_train_t = torch.tensor(X_num_train, dtype=torch.float32)
    X_cat_train_t = torch.tensor(X_cat_train, dtype=torch.long)
    X_num_test_t = torch.tensor(X_num_test, dtype=torch.float32)
    X_cat_test_t = torch.tensor(X_cat_test, dtype=torch.long)

    num_cols, cat_cols = X_num_train_t.shape[1], X_cat_train_t.shape[1]
    categories = [len(set(X_cat_train[:, i])) for i in range(X_cat_train.shape[1])]
    print(f"Num features: {num_cols}, Cat features: {cat_cols}, Categories: {categories}")

    # Crear DataLoaders (incluir etiquetas siempre si se va a hacer fine-tuning)
    train_ds = TensorDataset(X_num_train_t, X_cat_train_t, y_train_tensor)
    test_ds = TensorDataset(X_num_test_t, X_cat_test_t, y_test_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # =====================================
    # --- STAGE 1: VAE Pre-training ---
    # =====================================
    print("\n--- Starting Stage 1: VAE Pre-training ---")
    vae_model = ModelVAE(
        num_layers=NUM_LAYERS, d_numerical=num_cols, categories=categories, d_token=D_TOKEN,
        num_classes=None, # SIN cabeza clasificadora para pre-entrenamiento
        n_head=N_HEAD, factor=FACTOR, bias=TOKEN_BIAS
    )
    vae_model = vae_model.to(device)
    print(vae_model)

    optimizer_pt = torch.optim.AdamW(vae_model.parameters(), lr=LR_PRETRAIN, weight_decay=WD)
    scheduler_pt = ReduceLROnPlateau(optimizer_pt, mode='min', factor=0.95, patience=10, verbose=True) # Paciencia separada

    beta = MAX_BETA
    patience_counter = 0
    best_val_loss_pt = float('inf')

    for epoch in range(1, pretrain_epochs + 1):
        # Entrenar con alpha=0 (solo reconstrucción)
        train_mse, train_ce, train_kld, _, train_loss = \
            train_vae(vae_model, train_loader, optimizer_pt, device, beta, alpha=0.0)

        # Validar con alpha=0
        val_mse, val_ce, val_kld, _, val_loss = \
            validate_vae(vae_model, test_loader, device, beta, alpha=0.0)

        scheduler_pt.step(val_loss)

        if val_loss < best_val_loss_pt:
            print(f"Stage 1 - Epoch {epoch}: Val loss improved to {val_loss:.6f}. Saving pre-trained VAE...")
            best_val_loss_pt = val_loss
            torch.save(vae_model.state_dict(), pretrained_vae_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Stage 1 - Epoch {epoch}: Val loss did not improve ({val_loss:.6f} vs best {best_val_loss_pt:.6f}). Patience: {patience_counter}/{PATIENCE_PRETRAIN}")
            if patience_counter >= PATIENCE_PRETRAIN:
                 if beta > MIN_BETA:
                     beta = max(beta * LAMBDA_BETA, MIN_BETA)
                     print(f"Applied beta annealing. New beta: {beta:.6f}")
                     patience_counter = 0 # Resetear paciencia al cambiar beta
                 else:
                     print("Early stopping pre-training.")
                     break # Salir del bucle de pre-entrenamiento

        print(f"Stage 1 - Epoch {epoch:03d}/{pretrain_epochs} | Beta: {beta:.2e} | Loss: {train_loss:.4f} (MSE:{train_mse:.4f}, CE:{train_ce:.4f}, KLD:{train_kld:.4f}) | Val Loss: {val_loss:.4f} (MSE:{val_mse:.4f}, CE:{val_ce:.4f}, KLD:{val_kld:.4f})")

    print(f"--- Stage 1 Finished. Best pre-training validation loss: {best_val_loss_pt:.6f} ---")

    # =======================================
    # --- STAGE 2: Supervised Fine-tuning ---
    # =======================================
    if perform_fine_tune:
        print("\n--- Starting Stage 2: Supervised Fine-tuning ---")

        # Crear modelo CON cabeza clasificadora
        ft_model = ModelVAE(
            num_layers=NUM_LAYERS, d_numerical=num_cols, categories=categories, d_token=D_TOKEN,
            num_classes=num_classes, # AHORA con cabeza clasificadora
            n_head=N_HEAD, factor=FACTOR, bias=TOKEN_BIAS
        )
        ft_model = ft_model.to(device)
        print("Fine-tuning model architecture:")
        print(ft_model)


        # Cargar pesos pre-entrenados
        load_pretrained_weights(ft_model, pretrained_vae_path, device)

        # Nuevo optimizador para fine-tuning (todos los parámetros, LR más bajo)
        optimizer_ft = torch.optim.AdamW(ft_model.parameters(), lr=LR_FINETUNE, weight_decay=WD)
        scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.9, patience=5, verbose=True) # Menor paciencia para FT

        beta = MIN_BETA # Empezar con beta bajo o mantener el final del pre-entrenamiento? Aquí usamos MIN_BETA
        patience_counter = 0
        best_val_loss_ft = float('inf')

        for epoch in range(1, finetune_epochs + 1):
            # Entrenar con alpha > 0
            train_mse, train_ce, train_kld, train_class_loss, train_loss = \
                train_vae(ft_model, train_loader, optimizer_ft, device, beta, alpha=ALPHA_FT)

            # Validar con alpha > 0
            val_mse, val_ce, val_kld, val_class_loss, val_loss = \
                validate_vae(ft_model, test_loader, device, beta, alpha=ALPHA_FT)

            scheduler_ft.step(val_loss) # Usar scheduler de fine-tuning

            if val_loss < best_val_loss_ft:
                print(f"Stage 2 - Epoch {epoch}: Val loss improved to {val_loss:.6f}. Saving fine-tuned model...")
                best_val_loss_ft = val_loss
                torch.save(ft_model.state_dict(), finetuned_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Stage 2 - Epoch {epoch}: Val loss did not improve ({val_loss:.6f} vs best {best_val_loss_ft:.6f}). Patience: {patience_counter}/{PATIENCE_FINETUNE}")
                if patience_counter >= PATIENCE_FINETUNE:
                    print("Early stopping fine-tuning.")
                    break # Salir del bucle de fine-tuning

            print(f"Stage 2 - Epoch {epoch:03d}/{finetune_epochs} | Beta: {beta:.2e} | Alpha: {ALPHA_FT:.2f} | Loss: {train_loss:.4f} (MSE:{train_mse:.4f}, CE:{train_ce:.4f}, KLD:{train_kld:.4f}, ClassL:{train_class_loss:.4f}) | Val Loss: {val_loss:.4f} (MSE:{val_mse:.4f}, CE:{val_ce:.4f}, KLD:{val_kld:.4f}, ClassL:{val_class_loss:.4f})")

        print(f"--- Stage 2 Finished. Best fine-tuning validation loss: {best_val_loss_ft:.6f} ---")

    else:
        print("\n--- Skipping Stage 2: Supervised Fine-tuning (perform_fine_tune=False) ---")

    # ================================================
    # --- Guardado Final de Encoder/Decoder/Embeddings ---
    # ================================================
    print(f"\nLoading final model from: {final_model_path}")
    # Cargar el modelo final (pre-entrenado o fine-tuneado)
    final_model_state_dict = torch.load(final_model_path, map_location=device)
    # Necesitamos crear una instancia del modelo correspondiente al state_dict final
    final_model_instance = ModelVAE(
        num_layers=NUM_LAYERS, d_numerical=num_cols, categories=categories, d_token=D_TOKEN,
        num_classes=num_classes if perform_fine_tune else None, # Asegurar que la estructura coincida
        n_head=N_HEAD, factor=FACTOR, bias=TOKEN_BIAS
    ).to(device)
    final_model_instance.load_state_dict(final_model_state_dict)
    final_model_instance.eval()
    print("Final model loaded successfully.")


    # Crear instancias de Encoder/Decoder para guardarlos por separado
    pre_encoder = EncoderModel(NUM_LAYERS, num_cols, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)
    pre_decoder = DecoderModel(NUM_LAYERS, num_cols, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)

    try:
        # Cargar pesos desde el *modelo final* (instancia ya cargada)
        pre_encoder.load_weights(final_model_instance)
        pre_decoder.load_weights(final_model_instance)
        print("Weights loaded into final pre_encoder and pre_decoder.")

        encoder_save_path = os.path.join(ckpt_dir, 'encoder_final.pt')
        decoder_save_path = os.path.join(ckpt_dir, 'decoder_final.pt')
        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)
        print(f"Saved final encoder to {encoder_save_path} and decoder to {decoder_save_path}")

        # Guardar embeddings latentes de entrenamiento del *modelo final*
        with torch.no_grad():
             train_z_mu = final_model_instance.get_embedding(X_num_train_t.to(device), X_cat_train_t.to(device))
             train_z_cls = train_z_mu[:, 0, :].detach().cpu().numpy() # Usar CLS token
             np.save(os.path.join(ckpt_dir, 'train_z_final.npy'), train_z_cls)
             print(f"Saved final latent embeddings (CLS token) to {os.path.join(ckpt_dir, 'train_z_final.npy')}")

    except Exception as e:
        print(f"Warning: Could not save separate encoder/decoder weights or embeddings from final model due to error: {e}")


if __name__ == "__main__":
    dataset_path = '/mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/metadata.json' # Verifica esta ruta
    encoding = 'ordinal'
    random_state = 0
    pretrain_epochs_main = 75 # Más épocas para pre-entrenamiento
    finetune_epochs_main = 40 # Épocas para fine-tuning
    batch_size_main = 128
    ckpt_dir_ft = 'ckpt_custom_model_staged_ft' # Directorio para el fine-tuning en etapas
    ckpt_dir_no_ft = 'ckpt_custom_model_no_ft' # Directorio si no se hace fine-tuning

    # --- Ejecutar con Fine-Tuning ---
    print("="*30)
    print(" RUNNING WITH FINE-TUNING ".center(30, "="))
    print("="*30)
    main(
        dataset_path=dataset_path, encoding=encoding, random_state=random_state,
        batch_size=batch_size_main,
        pretrain_epochs=pretrain_epochs_main,
        finetune_epochs=finetune_epochs_main,
        ckpt_dir=ckpt_dir_ft,
        perform_fine_tune=True
    )

    # --- Ejecutar SIN Fine-Tuning ---
    # print("\n" + "="*30)
    # print(" RUNNING WITHOUT FINE-TUNING ".center(30, "="))
    # print("="*30)
    # main(
    #     dataset_path=dataset_path, encoding=encoding, random_state=random_state,
    #     batch_size=batch_size_main,
    #     pretrain_epochs=pretrain_epochs_main, # Solo se ejecuta esta fase
    #     finetune_epochs=0, # No relevante
    #     ckpt_dir=ckpt_dir_no_ft,
    #     perform_fine_tune=False
    # )