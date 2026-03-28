# SeqGAN with Pretrained RoBERTa (Semantic Tuning)

This project trains a **SeqGAN** to generate amino acid sequences.
In the **semantic tuning stage**, we load a **pretrained RoBERTa** model, freeze its weights, and use it to guide the SeqGAN training process.

---

## How It Works

1. **Input Data**

   * A CSV file with a column `iAAs` (amino acid sequences).
   * Sequences are tokenized into 22 symbols: `<PAD>`, `X`, and 20 amino acids.

2. **Pretrained RoBERTa**

   * Loaded from `./model/words_model_ep3.pth`.
   * Parameters are **frozen** (not updated).
   * Provides semantic embeddings for the SeqGAN generator.

3. **SeqGAN Training**

   * **Generator (G):** Uses RoBERTa embeddings to produce new sequences.
   * **Discriminator (D):** Learns to tell real sequences (from CSV) vs. generated ones.
   * **Training steps:**

     * Pretrain G with **MLE loss** (teacher-forcing).
     * Fine-tune with **adversarial loss** from D.

---

## Run Example

```bash
python train_seqgan_htfr1.py \
  --csv_path ../data/semantic_tuning_data.csv \
  --test_ratio 0.1 \
  --seed 42
```

---

## Outputs

* **Checkpoints**: saved under `../checkpoint/pretrain/train/.../G/` and `D/`
* **Figures**: training curves in `../Figures/PreGan/.../`
* **Logs**: `LOSS_ACC.txt`, `LOSS_ACC_TEST.txt`
