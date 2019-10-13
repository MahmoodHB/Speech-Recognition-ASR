# Generate mfccs from a time series

y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=5)
librosa.feature.mfcc(y=y, sr=sr)
# array([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
# [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
# ...,
# [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
# [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])

# Use a pre-computed log-power Mel spectrogram

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                   fmax=8000)
librosa.feature.mfcc(S=librosa.power_to_db(S))
# array([[ -5.207e+02,  -4.898e+02, ...,  -5.207e+02,  -5.207e+02],
# [ -2.576e-14,   4.054e+01, ...,  -3.997e-14,  -3.997e-14],
# ...,
# [  7.105e-15,  -3.534e+00, ...,   0.000e+00,   0.000e+00],
# [  3.020e-14,  -2.613e+00, ...,   3.553e-14,   3.553e-14]])

# Get more components

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# Visualize the MFCC series

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

# Compare different DCT bases

m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(m_slaney, x_axis='time')
plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
plt.colorbar()
plt.subplot(2, 1, 2)
librosa.display.specshow(m_htk, x_axis='time')
plt.title('HTK-style (dct_type=3)')
plt.colorbar()
plt.tight_layout()
