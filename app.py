import os
import streamlit as st
import torch
import librosa
import numpy as np
import torch.nn as nn

# --- 1. Replicate Model Architecture ---
class DeepfakeAudioClassifier(nn.Module):
    def __init__(self):
        super(DeepfakeAudioClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- 2. Cache Model Loading for Speed ---
@st.cache_resource
def load_detector_model(model_path='best_deepfake_detector.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeAudioClassifier()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        return None, device

# --- 3. Streamlit UI Setup ---
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")

st.title("AI Deepfake Audio Detection Engine")
st.markdown("Upload any standard `.wav`, `.mp3`, or `.flac` voice sample to evaluate whether the speech is a **Genuine Human Utterance** or an **AI-Generated Deepfake**.")
st.write("---")

# Load model weights smoothly
model, device = load_detector_model()

if model is None:
    st.error("Error: `best_deepfake_detector.pth` weight file missing from the root directory. Please upload your model file to continue.")
else:
    # File Uploader Interface Widget
    uploaded_file = st.file_uploader("Drop or choose your audio file sample...", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        st.write("### Signal Playback & Audit Analysis")
        # Visual playback bar
        st.audio(uploaded_file, format='audio/wav')
        
        # Action Processing Button
        if st.button("Run Algorithmic Forensic Audit", type="primary"):
            with st.spinner("Processing audio matrix arrays... please hold."):
                try:
                    # Pipeline Constraints
                    sr, duration = 16000, 3
                    target_length = sr * duration
                    
                    # Read the memory buffer file object directly into librosa
                    audio, _ = librosa.load(uploaded_file, sr=sr)
                    
                    # Normalize time dimension
                    if len(audio) < target_length:
                        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
                    else:
                        audio = audio[:target_length]

                    # Map Spectrogram Features
                    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Scale Grid Range [-1, 1]
                    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
                    mel_spec_db = (mel_spec_db * 2) - 1

                    # Tensor Translation
                    X = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    # Model Evaluation Frame
                    with torch.no_grad():
                        output = model(X).squeeze(1)
                        probability = torch.sigmoid(output).item()

                    # Calibrated Boundary Decision (0.5000 Probability matching your 0.0002 boundary)
                    threshold = 0.5000
                    
                    st.write("---")
                    st.subheader("System Forensic Verdict")
                    
                    if probability >= threshold:
                        st.error("**Verdict: DEEPFAKE (AI-Generated Speech)**")
                        confidence = probability
                    else:
                        st.success("**Verdict: GENUINE (Human Speech)**")
                        confidence = 1.0 - probability

                    # Beautiful Confidence Metrics Card
                    st.metric(label="Analysis Verification Confidence", value=f"{confidence * 100:.2f}%")
                    st.progress(confidence)
                    
                    # Hidden Metadata Dropdown for developer inspection
                    with st.expander("Show Raw Engineering Logits"):
                        st.text(f"Raw Sigmoid Output Float: {probability:.6f}")
                        st.text(f"Model Processing Device  : {str(device).upper()}")

                except Exception as e:
                    st.error(f"Core processing crash: {str(e)}")
