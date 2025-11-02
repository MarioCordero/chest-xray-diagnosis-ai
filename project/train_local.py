# train_local.py
from fastai.vision.all import *
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np, matplotlib.pyplot as plt, argparse, os

def main(data_dir, epochs, bs, num_workers):
    DATA_DIR = Path(data_dir)
    assert DATA_DIR.exists(), f"No se encontró {DATA_DIR}"
    set_seed(42, reproducible=True)

    # ----- Datasets & DataLoaders -----
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(train_name='train', valid_name='val'),
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224, min_scale=0.8),
                    Normalize.from_stats(*imagenet_stats)]
    )
    dls = block.dataloaders(DATA_DIR, bs=bs, num_workers=num_workers)
    print("Clases:", dls.vocab)

    # ----- Métricas online seguras (GPU): solo accuracy -----
    learn = vision_learner(dls, resnet18, metrics=[accuracy])

    # ----- Entrenamiento -----
    learn.fine_tune(epochs)

    os.makedirs("outputs", exist_ok=True)

    # =========================
    # VALIDACIÓN
    # =========================
    preds_val, targs_val = learn.get_preds(dl=learn.dls.valid)
    # y_true / y_pred
    y_true = targs_val.cpu().numpy()
    y_pred = preds_val.argmax(dim=1).cpu().numpy()

    print("\n--- VALID REPORT (sklearn) ---")
    print(classification_report(y_true, y_pred, target_names=dls.vocab))
    f1_macro_val = f1_score(y_true, y_pred, average='macro')
    print(f"F1 (macro) - valid: {f1_macro_val:.4f}")

    # Matriz de confusión (valid)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dls.vocab)
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title('Confusion Matrix – Validación')
    plt.tight_layout()
    plt.savefig("outputs/cm_val.png", dpi=180); plt.close()

    # ROC-AUC (binario): prob de clase 'PNEUMONIA'
    probs_val = torch.softmax(preds_val, dim=1)[:, dls.vocab.o2i['PNEUMONIA']].cpu().numpy()
    fpr, tpr, _ = roc_curve(y_true, probs_val)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_val:.3f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC – Validación')
    plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
    plt.savefig("outputs/roc_val.png", dpi=180); plt.close()
    print(f"AUC (valid): {roc_auc_val:.4f}")

    # =========================
    # TEST (si existe)
    # =========================
    TEST_DIR = DATA_DIR/'test'
    if TEST_DIR.exists():
        test_files = get_image_files(TEST_DIR)
        test_dl = learn.dls.test_dl(test_files)
        preds_test, _ = learn.get_preds(dl=test_dl)

        y_test = np.array([dls.vocab.o2i[parent_label(f)] for f in test_files])
        y_hat  = preds_test.argmax(dim=1).cpu().numpy()

        print("\n--- TEST REPORT (sklearn) ---")
        print(classification_report(y_test, y_hat, target_names=dls.vocab))
        f1_macro_test = f1_score(y_test, y_hat, average='macro')
        print(f"F1 (macro) - test: {f1_macro_test:.4f}")

        probs_test = torch.softmax(preds_test, dim=1)[:, dls.vocab.o2i['PNEUMONIA']].cpu().numpy()
        fpr_t, tpr_t, _ = roc_curve(y_test, probs_test)
        roc_auc_test = auc(fpr_t, tpr_t)
        plt.figure()
        plt.plot(fpr_t, tpr_t, label=f'Test ROC (AUC = {roc_auc_test:.3f})')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC – Test')
        plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
        plt.savefig("outputs/roc_test.png", dpi=180); plt.close()
        print(f"AUC (test): {roc_auc_test:.4f}")

    # ----- Export -----
    learn.export("outputs/export.pkl")
    print("\nListo. Artefactos en ./outputs : export.pkl, cm_val.png, roc_val.png, (roc_test.png si aplica)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Ruta a chest_xray/")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    main(args.data_dir, args.epochs, args.bs, args.num_workers)