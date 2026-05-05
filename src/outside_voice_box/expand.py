#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

    Sébastien Le Maguer <sebastien.lemaguer@helsinki.fi>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created:  5 May 2026
"""

# Core Python
import os
import pathlib
import warnings
import re
import json
import math
import argparse

# Messaging/logging
import logging
from logging.config import dictConfig

try:
    import pythonjsonlogger

    JSON_LOGGER = True
except Exception:
    JSON_LOGGER = False

# Data
import numpy as np
import torch
import torch.nn as nn
from g2p_en import G2p

# Audio
import torchaudio
from tronduo.hparams import create_hparams
import soundfile as sf

# HiFiGAN
from outside_voice_box.hifigan.env import AttrDict
from outside_voice_box.hifigan.models import Generator

# Tronduo
from outside_voice_box.tronduo.hifigandenoiser import Denoiser
from outside_voice_box.tronduo.model_util import load_model
from outside_voice_box.tronduo import text_to_sequence

###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]
DEVICE = "cuda"
MAX_WAV_VALUE = 32768.0

# Define some flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


###############################################################################
# Functions
###############################################################################
def configure_logger(args) -> logging.Logger:
    """Setup the global logging configurations and instanciate a specific logger for the current script

    Parameters
    ----------
    args : dict
        The arguments given to the script

    Returns
    --------
    the logger: logger.Logger
    """
    # create logger and formatter
    logger = logging.getLogger()

    # Verbose level => logging level
    log_level = args.verbosity
    if args.verbosity >= len(LEVEL):
        log_level = len(LEVEL) - 1
        # logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)

    # Define the default logger configuration
    logging_config = dict(
        version=1,
        disable_existing_logger=True,
        formatters={
            "f": {
                "format": "[%(asctime)s] [%(levelname)s] — [%(name)s — %(funcName)s:%(lineno)d] %(message)s",
                "datefmt": "%d/%b/%Y: %H:%M:%S ",
            }
        },
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": LEVEL[log_level],
            }
        },
        root={"handlers": ["h"], "level": LEVEL[log_level]},
    )

    # Add file handler if file logging required
    if args.log_file is not None:
        cur_formatter_key = "f"
        if JSON_LOGGER:
            logging_config["formatters"]["j"] = {
                "()": "pythonjsonlogger.json.JsonFormatter",
                "fmt": "%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s",
                "rename_fields": {"asctime": "time", "levelname": "level", "lineno": "line_number"},
            }
            cur_formatter_key = "j"

        logging_config["handlers"]["f"] = {
            "class": "logging.FileHandler",
            "formatter": cur_formatter_key,
            "level": LEVEL[log_level],
            "filename": args.log_file,
        }
        logging_config["root"]["handlers"] = ["h", "f"]

    # Setup logging configuration
    dictConfig(logging_config)

    # Retrieve and return the logger dedicated to the script
    logger = logging.getLogger(__name__)
    return logger


def define_argument_parser() -> argparse.ArgumentParser:
    """Defines the argument parser

    Returns
    --------
    The argument parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="")

    # Add logging options
    parser.add_argument("-l", "--log_file", default=None, help="Logger file")
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity",
    )

    # Add performative options
    parser.add_argument("-j", "--speaker1_weight", default=0.5, type=float, help="????")
    parser.add_argument("-k", "--speaker2_weight", default=0.5, type=float, help="????")
    parser.add_argument("-m", "--prosody_strength", default=0.2, type=float, help="????")

    # Add arguments
    parser.add_argument("tronduo_model_path", help="the path to the TRONDUO model")
    parser.add_argument("hifigan_model_path", help="the path to the HIFIGAN model")
    parser.add_argument("sgr_model_path", help="the path to the SGR2 model")
    parser.add_argument("text", help="The text to synthesize")
    parser.add_argument("output_file", help="The output wavfile")
    # TODO

    # Return parser
    return parser


###############################################################################
# Operational functions
###############################################################################
def init_torch():
    torch.random.manual_seed(0)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    return device


def load_checkpoint(filepath, device):
    print(f"Loading '{filepath}'")
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


###############################################################################
# Entry point
###############################################################################
def main():
    # Initialization of the argument parser and the logger
    arg_parser = define_argument_parser()
    args = arg_parser.parse_args()
    logger = configure_logger(args)

    device = init_torch()

    # Define
    g2p = G2p()

    # Tronduo: Define parameters and load the model (TODO maybe put this in the configuration?)
    tronduo_params = create_hparams()
    tronduo_params.global_mean = None
    tronduo_params.distributed_run = False
    tronduo_params.prosodic = True
    tronduo_params.speakers = True
    tronduo_params.feat_dim = 1
    tronduo_params.feat_max_bg = 8
    tronduo_params.n_speakers = 2
    tronduo_params.speaker_embedding_dim = 8

    tronduo_checkpoint_path = pathlib.Path(args.tronduo_model_path)
    model = load_model(tronduo_params)
    model.load_state_dict(torch.load(tronduo_checkpoint_path)["state_dict"])
    _ = model.cuda().eval().half()  # TODO: this should be generalized

    # Load hifigan
    hifigan_checkpoint_path = pathlib.Path(args.hifigan_model_path)
    hifigan_config_file = hifigan_checkpoint_path.parent / "config.json"
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

    ### Load SGR tool
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v2 = bundle.get_model().to(device)
    w2v2.eval()

    sgr2 = MLPClassifier()
    sgr2.to(device)
    sgr2.load_state_dict(torch.load(args.sgr_model_path))
    sgr2.eval()

    text = args.text
    txt = re.sub(r"[\!.?]+", "", text)
    txt = re.sub(r";", ".", txt)
    phon = g2p(txt)
    for j, n in enumerate(phon):
        if n == " ":
            phon[j] = "} {"
    transcript = "{ " + " ".join(phon) + " }."
    transcript = re.sub(r"(\s+){ , }(\s+)", ",", transcript)
    transcript = re.sub(r"(\s+)?{ . }(\s+)?", ";", transcript)
    # transcript = re.sub(r' ; ', ';', transcript)
    transcript = re.sub(r"{ ", "{", transcript)
    transcript = re.sub(r" }", "}", transcript)

    # variation on speaker on a grid
    sequence = np.array(text_to_sequence(transcript, ["english_cleaners"]))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    speaks = torch.as_tensor([args.speaker1_weight, args.speaker2_weight]).unsqueeze(0).cuda()
    pros = torch.as_tensor(args.prosody_strength).unsqueeze(0).half().cuda()
    _, mel_outputs_postnet, _, _ = model.inference(sequence, speaks=speaks, pros=pros)
    durat = 1000
    while durat > 890: # FIXME: huh? Why this?
        try:
            _, mel_outputs_postnet, _, _ = model.inference(sequence, speaks=speaks, pros=pros)
            durat = mel_outputs_postnet[0].size()[1]
        except:
            pass
    melfl = mel_outputs_postnet.float()
    y_g_hat = generator(melfl)
    audio = denoiser(y_g_hat[0], strength=0.01).squeeze().half()
    audio_out = audio.cpu().detach().numpy()

    # Compute some additional information (not sure why here :/)
    waveform = torchaudio.functional.resample(audio.unsqueeze(0).float(), 22050, bundle.sample_rate)
    feats, _ = w2v2.extract_features(waveform.clone().to(device))
    vec = torch.mean(feats[2], dim=1)
    outputs = sgr2(vec).detach().cpu().numpy()
    pf = 1 / (1 + math.exp(-outputs[0][0]))
    pm = 1 / (1 + math.exp(-outputs[0][1]))

    # results.append(
    #     {
    #         "Filename": filename,
    #         "f": 100 * j,
    #         "m": 100 * k,
    #         "f0_in": m,
    #         "dur": np.round(len(audio_out) / tronduo_params.sampling_rate, 3),
    #         "utt": filenames[i],
    #         "pr_f": pf / (pf + pm),
    #         "pr_m": pm / (pf + pm),
    #         "gap": abs(pf - pm) / (pf + pm),
    #     }
    # )

    output_file = pathlib.Path(args.output_file)
    if not output_file.parent.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)
    sf.write(
        output_file,
        audio_out.astype("float32"),
        tronduo_params.sampling_rate,
    )


###############################################################################
# Wrapping for directly calling the scripts
###############################################################################
if __name__ == "__main__":
    main()
