# Cavity-QED Transmission: Parameter Estimation from Spectra
import os, json, math, random, pathlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, Model

# Reproducibility
SEED = 23
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

try:
    if tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Using mixed precision.")
except Exception as e:
    print("AMP not enabled:", e)

# Simple input-output theory (one atom)
# Transmission ~ | κ_ext / (iΔ + κ/2 + g^2/(iΔ + γ/2)) |^2
# We simulate over detuning Δ around resonance.

def cavity_transmission(delta, g, kappa, gamma, kappa_ext=None):
    # All in angular frequency units (arbitrary consistent units)
    if kappa_ext is None:
        kappa_ext = 0.6*kappa
    denom = 1j*delta + kappa/2.0 + (g**2)/(1j*delta + gamma/2.0)
    t = kappa_ext / denom
    return np.abs(t)**2

def synth_spectrum(num_points=256):
    # Sample physically plausible ranges (arbitrary units)
    g     = np.random.uniform(0.5, 5.0)       # coupling
    kappa = np.random.uniform(1.0, 8.0)       # cavity linewidth
    gamma = np.random.uniform(0.2, 3.0)       # atomic linewidth
    # Detuning sweep
    span = np.random.uniform(6.0, 14.0)
    delta = np.linspace(-span, span, num_points)
    spec = cavity_transmission(delta, g, kappa, gamma)
    # Normalize and add noise
    spec = spec / (spec.max()+1e-9)
    snr = np.random.uniform(20, 40)  # dB
    p_sig = np.mean(spec**2)
    p_noise = p_sig / (10**(snr/10))
    spec_noisy = spec + np.random.normal(0, np.sqrt(p_noise), spec.shape)
    spec_noisy = np.clip(spec_noisy, 0.0, 1.0).astype(np.float32)
    return spec_noisy, np.array([g, kappa, gamma], dtype=np.float32), delta.astype(np.float32)

def make_dataset(num=12000, L=256):
    X = np.zeros((num, L, 1), dtype=np.float32)
    Y = np.zeros((num, 3), dtype=np.float32)  # [g, kappa, gamma]
    DELTA = np.zeros((num, L), dtype=np.float32)
    for i in range(num):
        s, p, d = synth_spectrum(L)
        X[i,:,0] = s
        Y[i,:] = p
        DELTA[i,:] = d
    return X, Y, DELTA

# Quick peek
s, p, d = synth_spectrum(256)
plt.figure(figsize=(5,3))
plt.plot(d, s)
plt.title(f"Spectrum (g={p[0]:.2f}, κ={p[1]:.2f}, γ={p[2]:.2f})")
plt.xlabel("Δ")
plt.ylabel("T")
plt.tight_layout()
plt.savefig("figure")
plt.show()


# Data

L = 256
N_TR, N_VA, N_TE = 9000, 1500, 1500
X, Y, DEL = make_dataset(N_TR+N_VA+N_TE, L)
X_tr, X_va, X_te = np.split(X, [N_TR, N_TR+N_VA])
Y_tr, Y_va, Y_te = np.split(Y, [N_TR, N_TR+N_VA])
D_tr, D_va, D_te = np.split(DEL, [N_TR, N_TR+N_VA])

BATCH=64; AUTOTUNE=tf.data.AUTOTUNE
ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr)).shuffle(4096, seed=SEED).batch(BATCH).prefetch(AUTOTUNE)
ds_va = tf.data.Dataset.from_tensor_slices((X_va, Y_va)).batch(BATCH).prefetch(AUTOTUNE)
ds_te = tf.data.Dataset.from_tensor_slices((X_te, Y_te)).batch(BATCH).prefetch(AUTOTUNE)


# Model: lightweight 1D CNN regressor

def build_cnn_1d(input_shape=(256,1)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, padding='same')(inp); x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv1D(32, 7, padding='same')(x);   x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding='same')(x); x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv1D(64, 5, padding='same')(x); x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same')(x); x = layers.LeakyReLU(0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(3, activation='linear', dtype='float32')(x)  # [g, kappa, gamma]
    return Model(inp, out)

model = build_cnn_1d((L,1))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss='mse', metrics=['mae'])

out_dir = pathlib.Path("artifacts_cavity_qed"); out_dir.mkdir(exist_ok=True)
ckpt = tf.keras.callbacks.ModelCheckpoint(str(out_dir/"best_cqed.keras"), monitor='val_loss', save_best_only=True)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

hist = model.fit(ds_tr, validation_data=ds_va, epochs=80, callbacks=[ckpt, es, rlr])
with open(out_dir/"history.json","w") as f:
    json.dump({k:[float(x) for x in v] for k,v in hist.history.items()}, f)


# Evaluation & Visualization

metrics = model.evaluate(ds_te, return_dict=True)
print(metrics)

# Show a few predicted vs true spectra parameters
xb, yb = next(iter(ds_te))
yp = model.predict(xb)
for i in range(3):
    gT,kT,gamT = yb[i].numpy()
    gP,kP,gamP = yp[i]
    plt.figure(figsize=(5,3))
    # Rebuild spectra from predictions and truth on the same detuning grid
    d = D_te[i]
    s_true = cavity_transmission(d, gT, kT, gamT)
    s_true /= s_true.max()+1e-9
    s_pred = cavity_transmission(d, gP, kP, gamP)
    s_pred /= s_pred.max()+1e-9
    plt.plot(d, xb[i,:,0], label="Noisy sim (input)", alpha=0.5)
    plt.plot(d, s_true, label=f"Truth (g={gT:.2f}, κ={kT:.2f}, γ={gamT:.2f})")
    plt.plot(d, s_pred, '--', label=f"Pred (g={gP:.2f}, κ={kP:.2f}, γ={gamP:.2f})")
    plt.legend()
    plt.xlabel("Detuning Δ")
    plt.ylabel("Transmission")
    plt.tight_layout()
    plt.savefig("figure")
    plt.show()