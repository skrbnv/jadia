from glob import glob
import os
import sys
from jadia import Jadia, Segments
import jadia_plot as plot
import torch
import warnings
from datetime import datetime
import pickle
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Pipeline
from pyannote.database.util import load_lab
from dotenv import load_dotenv

# Load Jadia
diarizator = Jadia(
    device=torch.device("cuda:0"),
    model="lite_v2",
    batch_size=64,
)

# load pyannote DER function
pyannoteDER = DiarizationErrorRate()

# dirs
audio_dir = "tests/assets"
ext = ".wav"
ref_rttm_dir = "tests/assets"
target_dir = "results/"

try:
    os.mkdir(target_dir)
    os.mkdir(target_dir + "/images")
except:
    pass


# config
COLLAPSE = True
COLLAPSE_TOLERANCE = 0.7
USE_VAD = True
ADJUST_TOLERANCE = 0.7
MIN_INTERVAL = 0.2
SMOOTH = False
FAST_FIT = False
SPECTROGRAM_STRIDE = 5
UPDATE_STRIDE = 7
CLUSTERING_METHOD = "CosKMeans"
WITH_CONFIDENCE = True

# Use pyannote?
RUN_PYANNOTE = True

if RUN_PYANNOTE is True:
    load_dotenv()
    config = os.environ.keys()
    if "PYANNOTE_TOKEN" not in config:
        raise Exception(
            "Unable to find environment variable PYANNOTE_TOKEN while RUN_PYANNOTE is True. You may add it to .env file"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ.get("PYANNOTE_TOKEN"),
    ).to(torch.device("cuda"))


def test_eval():

    # load audio files
    files = sorted(glob(audio_dir + f"/*{ext}"))
    stats = {}
    for audio_fpath in files:
        print(f"Processing {os.path.basename(audio_fpath)}")
        print(f"  Collapse: {COLLAPSE}, VAD: {USE_VAD}, Tolerance: {ADJUST_TOLERANCE}")
        print(f"  Min interval: {MIN_INTERVAL}, Smooth: {SMOOTH}, Fast fit: {FAST_FIT}")
        name = ".".join(os.path.basename(audio_fpath).split(".")[:-1])

        # if os.path.isfile(f"{images_dir}/{name}_{CLUSTERING_METHOD}.png"):
        #    continue

        ref_rttm_fpath = os.path.join(ref_rttm_dir, name) + ".rttm"
        try:
            reference = Segments().from_rttm_load(ref_rttm_fpath)
        except:
            continue
        num_voices = len(reference.speakers())
        diarizator.setup(num_voices=num_voices, with_confidence=True)

        # Compute predictions [Jadia]
        jst = datetime.now()
        diarizator.load_audio(filename=audio_fpath)
        predictions = diarizator.predict(fast_fit=True)
        segments = diarizator.predictions_to_segments(predictions)
        jet = datetime.now()

        if RUN_PYANNOTE:
            # Compute predictions [Pyannote]
            pst = datetime.now()
            pydia = pipeline(audio_fpath)
            pet = datetime.now()
            with open("pypred.rttm", "w") as f:
                pydia.write_rttm(f)

        segments.to_rttm_save(
            fpath=os.path.join(target_dir, name) + ".rttm", overwrite=True
        )
        reference.to_lab_save("ref.lab")
        segments.to_lab_save("hyp.lab")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ref = load_lab("ref.lab")
            hyp = load_lab("hyp.lab")
            der = pyannoteDER(ref, hyp, detailed=True)
            os.remove("ref.lab")
            os.remove("hyp.lab")

        if RUN_PYANNOTE:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                der2 = pyannoteDER(ref, pydia, detailed=True)
                # der2_ = DER(ref, pydia)

        # plot combined chart
        if RUN_PYANNOTE:
            segments2 = Segments()
            segments2.from_rttm_load("pypred.rttm")
            os.remove("pypred.rttm")
        else:
            segments2 = None

        plot.plot_predictions(
            predictions=predictions,
            segments=segments,
            segments2=segments2,
            filename=f"{target_dir}/images/{name}_{CLUSTERING_METHOD}.png",
            ground_truth=reference,
        )

        plot.plot_segments(
            pred=segments,
            ground_truth=reference,
            filename=f"{target_dir}/images/{name}_{CLUSTERING_METHOD}_iv.png",
        )

        print("-----------[RESULTS: Jadia]------------")
        print(f"       Confusion: {der['confusion']:.4f}")
        print(f"     False alarm: {der['false alarm']:.4f}")
        print(f"Missed detection: {der['missed detection']:.4f}")
        print(f"             DER: {der['diarization error rate']:.4f}")
        print(f"            Time: {(jet-jst).total_seconds()}")

        if RUN_PYANNOTE:
            print("-----------[RESULTS: Pyannote]------------")
            print(f"       Confusion: {der2['confusion']:.4f}")
            print(f"     False alarm: {der2['false alarm']:.4f}")
            print(f"Missed detection: {der2['missed detection']:.4f}")
            print(f"             DER: {der2['diarization error rate']:.4f}")
            print(f"            Time: {(pet-pst).total_seconds()}")

        stats[name] = {
            "confusion": {
                "Jadia": der["confusion"],
            },
            "false_alarm": {
                "Jadia": der["false alarm"],
            },
            "missed_detection": {
                "Jadia": der["missed detection"],
            },
            "der": {
                "Jadia": der["diarization error rate"],
            },
            "time": {
                "Jadia": (jet - jst).total_seconds() * 1000,
            },
        }
        if RUN_PYANNOTE:
            stats[name]["confusion"]["pyannote"] = der2["confusion"]
            stats[name]["false_alarm"]["pyannote"] = der2["false alarm"]
            stats[name]["missed_detection"]["pyannote"] = der2["missed detection"]
            stats[name]["der"]["pyannote"] = der2["diarization error rate"]
            stats[name]["time"]["pyannote"] = (pet - pst).total_seconds() * 1000

    with open(os.path.join(target_dir, "stats.pkl"), "wb") as fp:
        pickle.dump(stats, fp)


if __name__ == "__main__":
    test_eval()
