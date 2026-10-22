# DCGAN on Fashion-MNIST

A Deep Convolutional Generative Adversarial Network (DCGAN) implemented in Keras and trained on the **Fashion-MNIST** dataset (60,000 grayscale 28x28 images of clothing items). The notebook trains a Generator and a Discriminator, generates synthetic images, and evaluates the results using loss curves, discriminator accuracy and visual quality.

Notebook: `DCGAN_FashionMNIST.ipynb`

---

## 1. Requirements

- Python 3.9 - 3.12
- Internet connection on first run (Keras downloads the dataset automatically)
- A GPU is recommended (Google Colab free GPU works well), but CPU also works (slower)

### Libraries

| Library | Purpose |
|---|---|
| `tensorflow` | Deep learning framework (includes Keras and h5py) |
| `numpy` | Array and numeric operations |
| `matplotlib` | Plotting images and loss curves |
| `pandas` | Smoothing the loss curves (rolling mean) |
| `pydot` | Optional - only for the model architecture diagram |

### Install

```bash
pip install tensorflow numpy matplotlib pandas pydot
```

The first cell of the notebook runs the same command (`!pip install -q ...`). On Google Colab everything is already installed.

---

## 2. How to Run

**Google Colab (easiest)**
1. Upload `DCGAN_FashionMNIST.ipynb` to https://colab.research.google.com
2. Runtime -> Change runtime type -> select **GPU**
3. Runtime -> **Run all**

**Local (Jupyter)**
1. Install the libraries above
2. Start Jupyter: `jupyter notebook`
3. Open the notebook and choose Kernel -> **Restart & Run All**

Run the cells in order from top to bottom.

---

## 3. Notebook Structure

| Section | Description |
|---|---|
| Install libraries | Installs the required packages |
| Load and view dataset | Loads Fashion-MNIST and plots 49 sample images |
| Discriminator | CNN that classifies images as real or fake |
| Generator | Transposed-convolution network that maps a 100-dim noise vector to a 28x28 image |
| Combined GAN | Generator + frozen Discriminator, used to update the Generator |
| Training | Alternating updates of Discriminator and Generator, with loss tracking |
| Evaluation | Loss curves, discriminator accuracy, generated images, real-vs-fake comparison, latent interpolation, written analysis |

---

## 4. Model Details

**Dataset:** Fashion-MNIST, images scaled to [-1, 1]

**Discriminator**
- Input: 28x28x1
- 2 x [Conv2D(64, 3x3, stride 2) + LeakyReLU(0.2) + Dropout(0.4)]
- Flatten -> Dense(1, sigmoid)

**Generator**
- Input: 100-dim latent vector
- Dense(128 x 7 x 7) + LeakyReLU -> Reshape (7, 7, 128)
- 2 x [Conv2DTranspose(128, 4x4, stride 2) + LeakyReLU] -> 28x28
- Conv2D(1, 7x7, tanh) -> 28x28x1 output

**Training settings**
- Loss: binary cross-entropy
- Optimizer: Adam (learning rate 0.0002, beta_1 = 0.5)
- Batch size: 128 (64 real + 64 fake for the Discriminator)
- Epochs: 50 (change `n_epochs` in the training cell)
- Latent dimension: 100

---

## 5. Output Files

Created in the working directory while the notebook runs:

| File | Description |
|---|---|
| `generated_plot_e010.png`, `e020`, ... | Generated image grids every 10 epochs |
| `generator_model_010.h5`, ... | Saved Generator weights every 10 epochs |
| `loss_curves.png` | Generator and Discriminator loss curves |
| `discriminator_accuracy.png` | Discriminator accuracy on real vs fake images |
| `final_generated_images.png` | Final 7x7 grid of synthetic images |
| `discriminator_plot.png`, `generator_plot.png` | Model diagrams (only if pydot and graphviz are installed) |

---

## 6. How to Interpret the Results

**Loss curves**
- A healthy GAN shows both losses oscillating in a stable range. Discriminator loss is typically around 0.5 - 0.7 and Generator loss around 0.7 - 1.5.
- Losses are not expected to fall steadily. The two networks compete, so a stable balance is the sign of good training.
- Discriminator loss near 0 with a rapidly rising Generator loss means the Discriminator is winning (training failure).

**Discriminator accuracy**
- Accuracy on real and fake images staying near 50% (roughly 40 - 70%) means the two networks are balanced.

**Image quality**
- Early epochs give noise and blobs. Later epochs should show recognizable clothing items (shirts, trousers, shoes, bags).
- Check the grid for diversity. Many identical images indicate mode collapse.
- The interpolation plot should show smooth transitions between images.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run the install cell or `pip install tensorflow numpy matplotlib pandas pydot` |
| Training is very slow | Use a GPU runtime in Colab or reduce `n_epochs` (e.g. 20) |
| Dataset download fails | Check your internet connection and re-run the cell |
| `plot_model skipped` message | Harmless. Install `pydot` and Graphviz if you want the diagram |
| HDF5 / `.h5` warning when saving | Harmless warning from newer Keras versions |
| Cells give errors after changes | Use Kernel -> Restart & Run All so cells run in order |
