#!/usr/bin/env python
# coding: utf-8

# Core python
import os
import warnings
import re
import json
import math

# Data
import numpy as np
import torch
import torch.nn as nn
from g2p_en import G2p
import pandas as pd

# Audio
import torchaudio
from tronduo.hparams import create_hparams
from scipy.io import wavfile
import soundfile as sf

# HiFiGAN
from outside_voice_box.hifigan.env import AttrDict
from outside_voice_box.hifigan.models import Generator

# Tronduo
from outside_voice_box.tronduo.hifigandenoiser import Denoiser
from outside_voice_box.tronduo.model_util import load_model
from outside_voice_box.tronduo import text_to_sequence

# Some beautifying helpers
from tqdm import tqdm

######################################################################################################

# Define some flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# Define some helper constants
DEVICE = "cuda"
MAX_WAV_VALUE = 32768.0


######################################################################################################

def init_torch():
    torch.random.manual_seed(0)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    return device

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x



######################################################################################################
def main():


    device = init_torch()
    OUT_FOLTER = "syn/"

    # Define
    g2p = G2p()

    # Tronduo: Define parameters and load the model
    tronduo_params = create_hparams()
    tronduo_params.global_mean = None
    tronduo_params.distributed_run = False
    tronduo_params.prosodic = True
    tronduo_params.speakers= True
    tronduo_params.feat_dim = 1
    tronduo_params.feat_max_bg = 8
    tronduo_params.n_speakers= 2
    tronduo_params.speaker_embedding_dim = 8

    tronduo_path = "models/ambiguous_model/"  # speaker vector
    tronduo_checkpoint_iter = "60000"  # number of iterations trained in the folder above
    tronduo_checkpoint_path = tronduo_path + "checkpoint_" + tronduo_checkpoint_iter


    model = load_model(tronduo_params)
    model.load_state_dict(torch.load(tronduo_checkpoint_path)["state_dict"])
    _ = model.cuda().eval().half()



    # Load hifigan
    hifigan_path = "models/hifigan/"
    hifigan_checkpoint_iter = 3100000
    hifigan_config_file = hifigan_path + "config.json"
    hifigan_checkpoint_path = hifigan_path + "g_" + str(hifigan_checkpoint_iter).zfill(8)
    with open(hifigan_config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    hifigan_params = AttrDict(json_config)
    torch.manual_seed(hifigan_params.seed)
    generator = Generator(hifigan_params).to(device)
    state_dict_g = load_checkpoint(hifigan_checkpoint_path, device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()

    # from hifigandenoiser import Denoiser
    denoiser = Denoiser(generator, mode="zeros")  # The other mode is normal/zeros

    # ### Text preparation

    texts = [
        "Meanwhile spring had passed well into summer.",
        "They had come not to admire but to observe.",
        "There were other farmhouses nearby.",
        "Outside, only a handful of reporters remained.",
        "Coverage of primary literature will follow.",
        "The data are presented in lists and tables.",
        "A third volume remains to be published.",
        "At intervals an alumni directory is issued.",
        "Let us differentiate a few of these ideas.",
        "The system works as an impersonal mechanism.",
    ]


    filenames = [
        "si1905",
        "si1710",
        "si1664",
        "si1552",
        "si1445",
        "si1442",
        "si1440",
        "si1297",
        "si1182",
        "si1083",
    ]


    transcripts = [""] * len(texts)
    txt = [re.sub(r"[\!.?]+", "", tr) for tr in texts]
    txt = [re.sub(r";", ".", tr) for tr in txt]
    for i in range(len(texts)):
        phon = g2p(txt[i])
        for j, n in enumerate(phon):
            if n == " ":
                phon[j] = "} {"
        transcripts[i] = "{ " + " ".join(phon) + " }."
    transcripts = [re.sub(r"(\s+){ , }(\s+)", ",", tr) for tr in transcripts]
    transcripts = [re.sub(r"(\s+)?{ . }(\s+)?", ";", tr) for tr in transcripts]
    # transcripts = [re.sub(r' ; ', ';', tr) for tr in transcripts]
    transcripts = [re.sub(r"{ ", "{", tr) for tr in transcripts]
    transcripts = [re.sub(r" }", "}", tr) for tr in transcripts]


    ### Load SGR tool
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v2 = bundle.get_model().to(device)
    w2v2.eval()


    sgr2 = MLPClassifier()
    sgr2.to(device)
    sgr2.load_state_dict(torch.load("models/sgr/sgr2_model_1gpu.pt"))
    sgr2.eval()


    # ### Test a range of outcomes on a grid of inputs

    results = []
    out = True
    if out and not os.path.exists(OUT_FOLTER):
        os.makedirs(OUT_FOLTER)

    # variation on speaker on a grid
    for i in tqdm(range(2)):
        sequence = np.array(text_to_sequence(transcripts[i], ["english_cleaners"]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        for j in np.arange(0.0, 1.01, 0.1):
            for k in np.arange(0.0, 1.01, 0.1):
                for m in [-0.2, 0.2]:
                    speaks = torch.as_tensor([j, k]).unsqueeze(0).cuda()
                    pros = torch.as_tensor(m).unsqueeze(0).half().cuda()
                    _, mel_outputs_postnet, _, _ = model.inference(sequence, speaks=speaks, pros=pros)
                    durat = 1000
                    while durat > 890:
                        try:
                            _, mel_outputs_postnet, _, _ = model.inference(
                                sequence, speaks=speaks, pros=pros
                            )
                            durat = mel_outputs_postnet[0].size()[1]
                        except:
                            pass
                    melfl = mel_outputs_postnet.float()
                    y_g_hat = generator(melfl)
                    audio = denoiser(y_g_hat[0], strength=0.01).squeeze().half()
                    audio_out = audio.cpu().detach().numpy()
                    # generate output
                    filename = f"{filenames[i]}_F{int(round(100*j,0)):04}M{int(round(100*k,0)):04}f0_{int(round(100*m,0)):03}"
                    waveform = torchaudio.functional.resample(
                        audio.unsqueeze(0).float(), 22050, bundle.sample_rate
                    )
                    feats, _ = w2v2.extract_features(waveform.clone().to(device))
                    vec = torch.mean(feats[2], dim=1)
                    outputs = sgr2(vec).detach().cpu().numpy()
                    pf = 1 / (1 + math.exp(-outputs[0][0]))
                    pm = 1 / (1 + math.exp(-outputs[0][1]))

                    results.append(
                        {
                            "Filename": filename,
                            "f": 100 * j,
                            "m": 100 * k,
                            "f0_in": m,
                            "dur": np.round(len(audio_out) / tronduo_params.sampling_rate, 3),
                            "utt": filenames[i],
                            "pr_f": pf / (pf + pm),
                            "pr_m": pm / (pf + pm),
                            "gap": abs(pf - pm) / (pf + pm),
                        }
                    )
                    if out:
                        print(filename)
                        sf.write(
                            OUT_FOLTER + filename + ".wav",
                            audio_out.astype("float32"),
                            tronduo_params.sampling_rate,
                        )

    results = pd.DataFrame(results)
    results.to_csv("test_results.csv", sep="|")

if __name__ == "__main__":
    main()
