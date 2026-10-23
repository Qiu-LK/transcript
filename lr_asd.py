"""CPU adapter for the MIT-licensed LR-ASD pretrained active-speaker model."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHT = ROOT / "models" / "visual" / "lr_asd_pretrain_ava.model"
VENDOR = ROOT / "third_party" / "lr_asd"
MODEL = None


def available():
    if not WEIGHT.is_file():
        return False
    try:
        _allow_base_packages()
        import torch
        import scipy
        import python_speech_features
        import cv2
        return True
    except ImportError:
        return False


def _allow_base_packages():
    """The project venv inherits CPU Torch/SciPy installed with local Python."""
    base = Path(sys.base_prefix) / "Lib" / "site-packages"
    if base.is_dir() and str(base) not in sys.path:
        sys.path.append(str(base))


def _load():
    global MODEL
    if MODEL is not None:
        return MODEL
    _allow_base_packages()
    import torch
    sys.path.insert(0, str(VENDOR))
    from model.Model import ASD_Model
    network = ASD_Model().cpu().eval()
    state = torch.load(str(WEIGHT), map_location="cpu", weights_only=True)
    weights = {key.removeprefix("model."): value for key, value in state.items()
               if key.startswith("model.")}
    network.load_state_dict(weights, strict=True)
    classifier = torch.nn.Linear(128, 2).cpu().eval()
    classifier.load_state_dict({"weight": state["lossAV.FC.weight"],
                                "bias": state["lossAV.FC.bias"]})
    torch.set_num_threads(min(4, torch.get_num_threads()))
    MODEL = network, classifier
    return MODEL


def _mfcc(audio):
    """13 MFCCs on 25 ms / 10 ms frames, matching LR-ASD training input."""
    _allow_base_packages()
    import python_speech_features
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < 400:
        return np.zeros((0, 13), dtype=np.float32)
    pcm = np.asarray(np.clip(x * 32767, -32768, 32767), dtype=np.int16)
    return np.asarray(python_speech_features.mfcc(
        pcm, 16000, numcep=13, winlen=0.025, winstep=0.010), dtype=np.float32)


def score(audio, faces):
    """Return one audio-visual speaking logit per 25 Hz grayscale face frame."""
    _allow_base_packages()
    import torch
    import cv2
    if len(faces) < 10:
        return np.zeros(0, dtype=np.float32)
    mfcc = _mfcc(audio)
    count = min(len(faces), len(mfcc) // 4)
    if count < 10:
        return np.zeros(0, dtype=np.float32)
    network, classifier = _load()
    result = []
    with torch.inference_mode():
        for start in range(0, count, 50):
            end = min(count, start + 50)
            if end - start < 4:
                break
            a = torch.from_numpy(np.asarray(mfcc[start*4:end*4], dtype=np.float32))[None]
            v = torch.from_numpy(np.asarray(faces[start:end], dtype=np.float32))[None]
            output = network(a, v)[0]
            logits = classifier(output)
            result.extend((logits[:, 1] - logits[:, 0]).cpu().numpy().tolist())
    return np.asarray(result, dtype=np.float32)
