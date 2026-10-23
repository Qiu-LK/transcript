from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import tarfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models" / "faster-whisper-medium"
BASE = "https://modelscope.cn/models/Systran/faster-whisper-medium/resolve/master/"
FILES = {
    "config.json": (2257, "3622a2ddc41ec0e0fd4e68c13c6830f03b90c38d89aaad184de02c8c642cf807"),
    "model.bin": (1527906378, "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae"),
    "tokenizer.json": (2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
    "vocabulary.txt": (459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
}
SPEAKER_DIR = ROOT / "models" / "speaker-diarization"
SEGMENTATION_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_ARCHIVE = (6958444, "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488")
SEGMENTATION_MODEL = (1540506, "d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d")
EMBEDDING_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMBEDDING_MODEL = (39593761, "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b")
MULTILINGUAL_URL = "https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"
MULTILINGUAL_MODEL = (71441526, "bf1a75b9930474cf3389ef415e6e5d38ca96fea4a3a00f7e301d080a58ee2239")
LR_ASD_URL = "https://api.github.com/repos/Junhua-Liao/LR-ASD/contents/weight/pretrain_AVA.model?ref=main"
LR_ASD_MODEL = (3426337, "85e6c77fc981595234790d1e128ebb60352d37726b2445e0ef8891e2512fe9e3")


def valid(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == digest


def download(url, target, size, digest):
    if valid(target, size, digest):
        print(f"已有且校验通过：{target.name}")
        return
    pending = target.with_name(target.name + ".part")
    print(f"正在下载 {target.name}（{size / 1024 / 1024:.1f} MB）...", flush=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    if "api.github.com" in url:
        headers["Accept"] = "application/vnd.github.raw+json"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response, pending.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if not valid(pending, size, digest):
            raise RuntimeError(f"{target.name} 下载不完整或校验失败")
        pending.replace(target)
    except Exception:
        pending.unlink(missing_ok=True)
        raise


def install_speaker_models():
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    mirror = "https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/main/"
    for name, size, digest in (
        ("wespeaker_en_voxceleb_resnet34_LM.onnx", 26530550, "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"),
        ("wespeaker_zh_cnceleb_resnet34_LM.onnx", 26530548, "87d1d5068397f3792c730570b53d66cd8be1da7ea22dd04f5b6706d96a3cd168"),
    ):
        download(mirror + name, SPEAKER_DIR / name, size, digest)
    embedding = SPEAKER_DIR / "embedding.onnx"
    download(EMBEDDING_URL, embedding, *EMBEDDING_MODEL)
    download(MULTILINGUAL_URL, SPEAKER_DIR / "embedding-eres2netv2.onnx", *MULTILINGUAL_MODEL)

    segmentation = SPEAKER_DIR / "segmentation.onnx"
    if valid(segmentation, *SEGMENTATION_MODEL):
        print("已有且校验通过：segmentation.onnx")
        return
    archive = SPEAKER_DIR / "segmentation.tar.bz2"
    download(SEGMENTATION_URL, archive, *SEGMENTATION_ARCHIVE)
    member_name = "sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
    pending = SPEAKER_DIR / "segmentation.onnx.part"
    try:
        with tarfile.open(archive, "r:bz2") as package:
            source = package.extractfile(member_name)
            if source is None:
                raise RuntimeError("说话人分离模型压缩包内容异常")
            with source, pending.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
        if not valid(pending, *SEGMENTATION_MODEL):
            raise RuntimeError("说话人分离模型解压后校验失败")
        pending.replace(segmentation)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    finally:
        archive.unlink(missing_ok=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        target = DEST / name
        download(BASE + name, target, size, digest)
    install_speaker_models()
    visual = ROOT / "models" / "visual"
    visual.mkdir(parents=True, exist_ok=True)
    download(LR_ASD_URL, visual / "lr_asd_pretrain_ava.model", *LR_ASD_MODEL)
    print("转写和说话人识别模型已准备好。")


if __name__ == "__main__":
    main()
