# GSoC 2025 FASEROH Task: Taylor Expansion Prediction

## Introduction

This repository contains my submission for the GSoC 2025 FASEROH evaluation test. The primary goal of this project is to explore the capability of sequence-to-sequence deep learning models in learning symbolic mathematical operations, specifically focusing on predicting the Taylor series expansion (up to the fourth order) of given mathematical functions.

The project involved three main tasks:

1.  **Dataset Generation & Preprocessing:** Creating a dataset of mathematical functions and their corresponding Taylor expansions using SymPy, followed by tokenization.
2.  **LSTM Model Training:** Implementing and training an LSTM-based sequence-to-sequence model with attention to learn the mapping from a function's token sequence to its Taylor expansion's token sequence.
3.  **Transformer Model Training (Optional):** Implementing and training a Transformer-based sequence-to-sequence model, exploring novel tokenization and attention mechanisms, to perform the same task.

This README provides a detailed overview of the implementation for each task, highlighting the methodologies, architectural choices, and key findings.
The images of model architectures are also uploaded alongside in this Github repository

## Repository Structure

The submission consists of three main Jupyter Notebooks:

1.  `Data_Generation.ipynb`: Contains the code for generating the dataset of functions and their Taylor expansions using SymPy and multiprocessing.
2.  `LSTM_Training.ipynb`: Details the implementation, training, and evaluation of the LSTM-based sequence-to-sequence model.
3.  `Transformer_Training.ipynb`: Details the implementation, training, and evaluation of the Transformer-based sequence-to-sequence model, including novel tokenization and attention mechanisms.

PDF versions of each notebook with outputs are also included.

---

## Task 1: Dataset Generation & Preprocessing (`Data_Generation.ipynb`)

### Objective

To programmatically generate a diverse dataset of mathematical functions represented as token sequences and their corresponding Taylor series expansions (up to the 4th order) around x=0.

### Methodology

1.  **Vocabulary Definition:** A vocabulary of mathematical operators (binary: `add`, `sub`, `mul`, `div`, `pow`; unary: `sq`, `sqrt`, `cb`, `cbrt`, `exp`, `ln`, `sin`, `cos`, etc.) and the variable `x` was defined. Each token was assigned a probability weight to influence its frequency during generation and an arity (0, 1, or 2).

2.  **Recursive Expression Generation (`generate_expression`):**
    *   A recursive function generates expressions in prefix notation (e.g., `['add', 'x', 'sin', 'x']` for `x + sin(x)`).
    *   The generation process respects operator arity and maximum tree depth (`max_depth=3` used for this submission to manage complexity).
    *   **Constraint:** To enhance diversity and avoid trivial cancellations or overly complex nests, if a function from a specific family (e.g., trigonometric, hyperbolic, exp/log, powers/roots) was chosen, subsequent recursive calls for its children were made using a vocabulary *excluding* other members of that *same* family within that branch.

3.  **SymPy Conversion (`sequence_to_sympy`):** The generated token sequence (prefix notation) is parsed recursively to construct the corresponding SymPy symbolic expression.

4.  **Taylor Expansion Calculation (`taylor_expansion`):**
    *   For each SymPy function, the Taylor expansion up to the 4th order around `x=0` was computed.
    *   SymPy's `diff` and `subs` (or `limit` if `subs` fails due to singularities at x=0) were used to calculate terms iteratively. `evalf(precision)` was used for numerical evaluation.
    *   A fallback to SymPy's `series().removeO()` was implemented for robustness, though the primary method involves manual term calculation.

5.  **Validation & Filtering:** Generated samples were validated to ensure suitability:
    *   **Real Coefficients (`not_real`):** Taylor expansions with complex coefficients were discarded.
    *   **Non-Constant (`is_constant`):** Constant functions/expansions were discarded.
    *   **Non-NaN:** Samples resulting in `sp.nan` (due to irrecoverable errors or undefined limits) were discarded.

6.  **Efficient Parallel Generation (`generate_dataset`, `generate_sample_with_timeout`):**
    *   Python's `multiprocessing` (with `set_start_method('fork')`) and `concurrent.futures.ProcessPoolExecutor` were employed to parallelize the generation process across multiple CPU cores (`n_workers=16`). This significantly sped up dataset creation.
    *   A timeout mechanism (`generate_sample_with_timeout`) was implemented for each sample generation to prevent SymPy from getting stuck on overly complex or problematic expressions, ensuring the overall process didn't stall.
    *   A progress bar (`tqdm`) tracked the number of valid samples generated versus total attempts, providing insight into the filtering rate.

7.  **Final Processing:** The generated data (function token list, SymPy function, SymPy Taylor expansion) was stored in a Pandas DataFrame. SymPy's `simplify` was applied to the input functions to create a canonical representation, and duplicates based on this simplified form were removed. The final DataFrame was saved using `pickle`.

### Challenges & Compute Constraints

Generating symbolic expressions and computing Taylor series, especially with validation checks, is computationally intensive. Even with multiprocessing on 16 cores, generating the dataset of 3000 valid, unique samples (after filtering) took approximately **3 hours** on my available hardware. This limitation restricted the feasible dataset size for this evaluation. A larger dataset would likely benefit model training further.

---

## Task 2: LSTM Model (`LSTM_Training.ipynb`)

### Objective

To train a sequence-to-sequence LSTM model with attention to predict the tokenized Taylor expansion from a tokenized input function.

### Tokenization (`Tokenizer` Class)

*   **Encoder Vocabulary:** Derived from the unique non-numeric tokens present in the *simplified* input functions generated in Task 1, plus special tokens (`PAD`, `<UNK>`) and common numeric components (`+`, `-`, `0`-`9`, `E-1` to `E1`).
*   **Decoder Vocabulary:** A fixed vocabulary designed to represent polynomial coefficients up to O(x^4). It includes:
    *   Special tokens: `PAD`, `SOS`, `EOS`.
    *   Coefficient degrees: `x0`, `x1`, `x2`, `x3`, `x4`.
    *   P10 Number Representation: Signs (`+`, `-`), Exponents (`E-5` to `E5`), Digits (`0`-`9`).
*   **P10 Number Encoding/Decoding:** A custom scheme (`encode_number`, `decode_number`) represents floating-point coefficients. It encodes a number into a fixed-length sequence of tokens (sign, exponent `E{exp}`, leading digit, `precision` mantissa digits). This provides a consistent token representation for numerical values. `precision=4` was used.
*   **Input Encoding (`encode_enc`):** Converts a simplified SymPy function into a sequence of encoder token IDs using `sympy_tokenizer` and `encode_number` for any numerical constants.
*   **Output Encoding (`encode_dec`):** Extracts coefficients of the Taylor polynomial (up to x^4) and converts them into a sequence of decoder token IDs, including `SOS`, `EOS`, degree markers (`x{i}`), and P10 encoded coefficients.
*   **Output Decoding (`decode_dec`, `seq_to_coeffs`):** Converts a sequence of predicted decoder token IDs back into a list of polynomial coefficients.

### Dataset & Dataloader

*   A custom PyTorch `Dataset` (`TaylorDataset`) was created to serve tokenized input/output pairs.
*   A custom `collate_fn` handles batching, padding input and output sequences to the maximum length within each batch using the `PAD` token ID (0). It also prepares input lengths required for `pack_padded_sequence`.

### Model Architecture: LSTM Seq2Seq with Attention

A standard Encoder-Decoder architecture with Bahdanau-style attention was implemented:

1.  **Encoder (`EncoderLSTM`):**
    *   An `nn.Embedding` layer maps input token IDs to `ENC_EMBED_DIM`-dimensional vectors.
    *   Dropout is applied to embeddings.
    *   An `nn.LSTM` (multi-layer, `N_LAYERS=2`) processes the padded sequences. `pack_padded_sequence` is used before the LSTM and `pad_packed_sequence` after to handle variable-length inputs efficiently.
    *   Outputs the LSTM's output sequence (for attention) and the final hidden and cell states (to initialize the decoder).

2.  **Attention (`Attention`):**
    *   A feed-forward network calculates attention scores (energy) based on the previous decoder hidden state and all encoder outputs.
    *   A `tanh` activation is used.
    *   A final linear layer (`v`) reduces the energy to a single score per encoder timestep.
    *   `softmax` converts scores to attention weights.
    *   Input padding mask is applied to prevent attention over `<PAD>` tokens.

3.  **Decoder (`DecoderLSTM`):**
    *   An `nn.Embedding` layer maps target token IDs to `DEC_EMBED_DIM`-dimensional vectors.
    *   Dropout is applied.
    *   At each timestep `t`:
        *   Takes the *previous* predicted token's embedding (`input`), the previous hidden/cell states, and all encoder outputs.
        *   Calculates attention weights using the `Attention` module based on the *current* hidden state and encoder outputs.
        *   Computes a weighted context vector by multiplying attention weights with encoder outputs.
        *   Concatenates the input embedding and the context vector.
        *   Feeds this concatenated vector into the `nn.LSTM` (multi-layer, `N_LAYERS=2`), using the previous hidden/cell states.
        *   The LSTM outputs the new hidden/cell states and an output vector.
        *   A final linear layer (`fc_out`) projects the concatenation of the LSTM output, the context vector, and the input embedding to the size of the decoder vocabulary, producing logits.

4.  **Seq2Seq Wrapper (`LSTMSeq2Seq`):**
    *   Combines the Encoder and Decoder.
    *   Implements the forward pass, handling the flow from encoder to decoder.
    *   Includes **Teacher Forcing** during training: With probability `teacher_forcing_ratio`, the true target token is fed as the next input; otherwise, the decoder's own prediction (argmax of logits) is used. Teacher forcing is disabled during validation/evaluation.
    *   Creates the source padding mask.

### Training

*   **Loss Function:** `nn.CrossEntropyLoss` with `ignore_index` set to the `PAD` token ID. **Weighted Cross Entropy** was used (via the `weight` argument), assigning higher weights to less frequent but important tokens (like degree markers `x{i}` and exponents `E{i}`) to encourage the model to predict them correctly. Weights were defined in the `Tokenizer`.
*   **Optimizer:** `AdamW` (Adam with decoupled weight decay, `LEARNING_RATE=1e-4`, `WEIGHT_DECAY=0.01`).
*   **Scheduler:** `CosineAnnealingLR` (`T_max=50`) to cyclically adjust the learning rate.
*   **Gradient Clipping:** Applied (`clip=1.0`) to prevent exploding gradients.
*   **Training Loop (`train_one_epoch`):** Standard PyTorch training loop iterating through batches, calculating loss, backpropagating, and updating weights.
*   **Validation Loop (`validate`):** Evaluates the model on the validation set after each epoch *without* teacher forcing.
*   **Early Stopping:** Training stops if the validation loss doesn't improve for a `patience` number of epochs (10 used here). The best model state dict based on validation loss is saved.

### Evaluation

*   **Inference (`evaluate_model`):**
    *   The trained model (best checkpoint loaded) generates predictions autoregressively on the test set.
    *   For each input function, the encoder runs once.
    *   The decoder starts with the `SOS` token. At each step, it predicts the next token based on its previous output.
    *   **Finite State Prediction (`finite_state_prediction`):** To ensure syntactically valid output sequences (e.g., a sign must follow a degree marker, digits must follow an exponent), a simple Finite State Machine (FSM) logic was added during inference. Based on the expected token type (degree, sign, exponent, number digit), the prediction is constrained to only valid tokens for that state by selecting the `argmax` only from the logits corresponding to valid next tokens.
    *   Decoding stops when `EOS` is predicted or a maximum length is reached.
*   **Metrics:**
    *   **Coefficient RMSE (`coeff_rmse`):** Calculates the Root Mean Squared Error between the predicted and target coefficient vectors.
    *   **Polynomial RMSE (`polynomial_rmse`):** Samples `n=100` points in the range `x_range=(-1, 1)`, evaluates the predicted and target polynomials at these points, and calculates the RMSE between the function values. This measures the functional similarity.

### Results

The trained LSTM model achieved:
*   Polynomial RMSE: ~323.21
*   Coefficient RMSE: ~315.13

The relatively high RMSE suggests the LSTM model struggled to achieve high precision, potentially due to the complexity of the symbolic mapping, the limitations of the dataset size, or the inherent challenges LSTMs face with very long-range dependencies compared to Transformers. The Finite State Prediction helps generate valid sequences but doesn't guarantee numerical accuracy.

---

## Task 3: Transformer Model (`Transformer_Training.ipynb`)

### Objective

To implement and train a Transformer model for the same Taylor expansion prediction task, leveraging its strengths in handling sequential data and exploring novel approaches to tokenization and positional information encoding.

### Novel Tokenization & Positional Embeddings (`Tokenizer` Class - Enhanced)

This is a key innovation in this implementation, designed to explicitly provide the Transformer with structural information about the input function tree and the numerical structure of the output polynomial.

*   **Encoder Tokenization (`sympy_tokenizer` modified):**
    *   Still uses `sympy_tokenizer` to get prefix notation tokens.
    *   **Crucially, it *also* generates an "absolute path" for each token in the expression tree.** This path represents the traversal from the root. For example, in `add(sin(x), mul(x, x))` which tokenizes to `['add', 'sin', 'x', 'mul', 'x', 'x']`:
        *   `'add'`: `[1]` (root)
        *   `'sin'`: `[1, 0, 1]` (first child of root)
        *   `'x' (in sin)`: `[1, 0, 1, 0]` (first child of 'sin')
        *   `'mul'`: `[1, 0, 1]` (second child of root)
        *   `'x' (first in mul)`: `[1, 0, 1, 1, 0]` (first child of 'mul')
        *   `'x' (second in mul)`: `[1, 0, 1, 0, 1]` (second child of 'mul')
    *   Numerical constants are still encoded using P10 (`encode_number`), but their corresponding paths are adapted (details in code). The paths are padded to a fixed `pos_dim=6`.

*   **Encoder Positional Embeddings (`encode_enc`):**
    *   **Absolute (`enc_a`):** The generated tree paths (padded vectors) are used directly as absolute positional encodings. This explicitly tells the model *where* each token resides within the function's structure.
    *   **Relative (`enc_r`):** For every pair of tokens `(i, j)` in the input sequence, a relative positional encoding is calculated by comparing their absolute path vectors element-wise using a simple difference function (`rel_ij`: returns -1 if `i=0, j>0`, 1 if `i>0, j=0`, 0 otherwise, element-wise). This results in a `(seq_len, seq_len, pos_dim)` tensor, capturing the *relationship* between the positions of any two tokens in the tree structure.

*   **Decoder Tokenization (`encode_dec`):**
    *   Similar to the LSTM decoder (P10 encoding, `SOS`, `EOS`, `x{i}` tokens).

*   **Decoder Positional Embeddings (`encode_dec`, `return_dec_embeddings`):**
    *   **Absolute (`dec_a`):** A simpler absolute encoding is used. Tokens belonging to the same coefficient (`x{i}`, sign, exponent, digits) share the same absolute position index (based on the coefficient's term number). This groups related tokens. Represented as a `(seq_len, 1)` tensor.
    *   **Relative (`dec_r`):** Calculated based on the *fine-grained position within a number*. Tokens representing digits closer to the decimal point have different relative encodings compared to those further away. This captures the numerical significance within the P10 encoding. Represented as a `(seq_len, seq_len, 1)` tensor.

*   **Pros of this Approach:**
    *   **Structure Awareness:** Encoder embeddings directly encode the tree structure, potentially allowing the Transformer to learn hierarchical relationships better than standard sequential embeddings.
    *   **Disentangled Information:** Relative embeddings capture pairwise relationships, complementing the absolute position information.
    *   **Tailored Embeddings:** Different, task-appropriate positional encoding strategies are used for the structured input (function tree) and the more sequentially structured output (polynomial coefficients).
    *   **Hypothesized Benefit:** This richer positional information, especially for the encoder, is hypothesized to allow the Transformer to more effectively understand the symbolic structure and perform the transformation to a Taylor series.

### Model Architecture: Transformer with Disentangled Attention

*   **Disentangled Attention (`DisentangledAttention`):**
    *   **Core Idea:** This is another novel component. Instead of adding positional encodings to word embeddings before attention, this custom attention mechanism incorporates positional information *directly* into the attention score calculation, keeping content and position somewhat separate (disentangled).
    *   **Implementation:** Inspired by papers like DeBERTa and implementations like x-transformers, this module modifies the standard scaled dot-product attention formula:
        `Attention(Q, K, V) = Softmax( (Qc*Kc^T + Qc*Kr^T + Kc*Qr^T + Pa*Pa^T) / sqrt(d) ) * V` (Conceptual formula; actual implementation uses efficient einsum/matmul and separate projections for content and position).
        *   `Qc, Kc`: Content queries and keys (from token embeddings).
        *   `Qr, Kr`: Relative position queries and keys (derived from relative positional encodings `enc_r`/`dec_r`).
        *   `Pa`: Absolute position embeddings (`enc_a`/`dec_a`).
    *   It takes separate inputs for content (`query`, `key`, `value`) and positional information (`a_query`, `a_key`, `r_query`, `r_key`).
    *   It uses separate linear projections for content and positional inputs before computing the attention scores.
    *   It supports causal masking for the decoder's self-attention.

*   **Encoder Block (`Encoder`):**
    *   Consists of a `DisentangledAttention` layer (for self-attention, using `enc_a` and `enc_r`) followed by Layer Normalization.
    *   A standard Feed-Forward Network (FFN) with ReLU activation, followed by Layer Normalization.
    *   Residual connections are used around both sub-layers.

*   **Decoder Block (`Decoder`):**
    *   Consists of a **masked** `DisentangledAttention` layer (for self-attention, using `dec_a` and `dec_r`) followed by Layer Normalization.
    *   A second `DisentangledAttention` layer (for cross-attention) attends to the encoder's output (`enc_x`), using the decoder's state (`dec_x`) as query and incorporating both encoder and decoder positional information (`enc_a`, `dec_a`, `enc_r`, `dec_r`), followed by Layer Normalization.
    *   A standard FFN, followed by Layer Normalization.
    *   Residual connections are used around all three sub-layers.

*   **Full Transformer (`Transformer`):**
    *   Input and output embedding layers (`enc_embed`, `dec_embed`).
    *   Stacks `N` Encoder blocks (`n_encoders=12`).
    *   Stacks `N` Decoder blocks (`n_decoders=12`).
    *   A final linear layer projects the decoder output to the decoder vocabulary size.

### Dataset & Dataloader

*   Uses the same `TaylorDataset` but with the enhanced tokenizer providing positional encodings.
*   The `collate_fn` is more complex, needing to pad:
    *   `enc_seqs`, `out_seqs`: 1D padding (standard).
    *   `enc_as`, `dec_as`: 2D padding (sequence length, position dimension).
    *   `enc_rs`, `dec_rs`: 3D padding (batch, seq_len, seq_len, position dimension) using custom helper functions (`pad_3d_tensors`, `pad_2d_tensors`).

### Training

*   Similar setup to the LSTM:
    *   **Loss:** Weighted `nn.CrossEntropyLoss` (same weights, ignoring PAD).
    *   **Optimizer:** `AdamW` (`lr=5e-5`, `weight_decay=0.01`).
    *   **Scheduler:** `CosineAnnealingWarmRestarts` (`T_0=10`, `T_mult=2`).
    *   **Training/Validation Loops:** Adapted for the Transformer architecture and the complex batch structure (multiple tensors for sequences and positions). Validation uses teacher forcing for simplicity and speed, primarily serving as a check against overfitting rather than a full generative evaluation during training.
    *   **Early Stopping:** Based on validation loss (`patience=10`).

### Evaluation

*   **Inference (`evaluate_model`):**
    *   Autoregressive generation similar to the LSTM evaluation.
    *   At each step, the required absolute (`dec_a`) and relative (`dec_r`) positional embeddings for the *current* prediction sequence length are generated on-the-fly using `tokenizer.return_dec_embeddings`.
    *   The **Finite State Prediction** (`finite_state_prediction`) logic is also used here during decoding to ensure syntactically valid coefficient sequences.
*   **Metrics:** Polynomial RMSE and Coefficient RMSE.

### Results

The trained Transformer model achieved:
*   Polynomial RMSE: ~15.67
*   Coefficient RMSE: ~18.26

These results show a significant improvement over the LSTM model, suggesting that the Transformer architecture, potentially aided by the novel structural tokenization and disentangled attention mechanism, is better suited for this symbolic task. The RMSE values are considerably lower, indicating better functional and coefficient-wise accuracy.

---

## Challenges, Limitations, and Future Work

*   **Compute Constraints:** The primary limitation was the lack of extensive GPU resources and time.
    *   This restricted the dataset size (3000 samples). A much larger dataset (e.g., 100k or 1M samples) would be ideal.
    *   It limited the extent of hyperparameter tuning (embedding dimensions, number of heads, layers, learning rates, dropout). The chosen parameters (`embed_dim=768`, `n_heads=4`, `n_layers=12`) are relatively large and might be suboptimal or prone to overfitting on the small dataset.
    *   Training duration was limited. The models, especially the Transformer, could likely benefit from significantly more training epochs.
*   **Positional Embedding/Attention Validation:** While the novel positional encoding and disentangled attention schemes implemented in the Transformer show promise (indicated by the improved RMSE), a more rigorous ablation study is needed to definitively isolate their contribution. Training baseline Transformers (e.g., with standard sinusoidal embeddings or learned embeddings only) on the same dataset would provide a clearer comparison. This was not feasible given the time constraints.
*   **Tokenization Granularity:** The P10 number representation, while functional, might not be the optimal way to handle coefficients. Exploring alternative numerical representations could be beneficial.
*   **Evaluation Metrics:** While RMSE provides a good measure, exploring other symbolic-specific metrics (e.g., exact match percentage after simplification, structural similarity metrics) could offer further insights.

Future work could involve:
1.  Generating a much larger dataset.
2.  Performing extensive hyperparameter optimization (e.g., using Optuna or Ray Tune).
3.  Conducting ablation studies on the Transformer's positional encoding and attention mechanisms.
4.  Training for longer durations.
5.  Exploring different numerical tokenization strategies.

---

## Conclusion

This project successfully demonstrated the feasibility of using sequence-to-sequence models for symbolic mathematics tasks like Taylor expansion prediction.

*   A robust data generation pipeline using SymPy and multiprocessing was established.
*   An LSTM with attention model was implemented and trained, providing a baseline performance.
*   A Transformer model incorporating novel structural positional encodings and a disentangled attention mechanism was developed and trained, achieving significantly better results than the LSTM on the generated dataset.

The promising results of the Transformer, despite compute limitations, highlight the potential of tailored architectures and input representations for symbolic reasoning tasks. I am enthusiastic about the potential of this research direction and eager to contribute further to the FASEROH project if selected for GSoC 2025.
