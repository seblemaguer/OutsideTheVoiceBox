#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

    Sébastien Le Maguer <sebastien.lemaguer@helsinki.fi>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created:  4 May 2026
"""

# Core Python
import pathlib
import argparse

# Messaging/logging
import logging
from logging.config import dictConfig

try:
    import pythonjsonlogger

    JSON_LOGGER = True
except Exception:
    JSON_LOGGER = False

import numpy as np
from scipy.io.wavfile import write as wave_write

import torch
import torch.nn.functional as F

from outside_voice_box.model import setup_tts_model, download_model_by_name, load_config


###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]


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
    parser.add_argument(
        "-D",
        "--dump_individual",
        action="store_true",
        help="Dump each intermediate reference weighted file (in the same directory than the final output file)",
    )
    parser.add_argument("-I", "--inverse", action="store_true", help="Inverse ombre blending")

    # Add arguments
    parser.add_argument("model_dir", help="The directory of themodel")
    parser.add_argument("text", help="The text to synthesize")
    parser.add_argument("output_file", help="The output wav file")
    # TODO

    # Return parser
    return parser


###############################################################################
# Operational functions
###############################################################################


# function to time align latents and transition from one to the other over time
def blend_latents_numpy(a_np, b_np) -> torch.Tensor:
    # Convert from numpy to torch tensors
    a = torch.from_numpy(a_np).float().unsqueeze(0)  # [1, T_a, 1024]
    b = torch.from_numpy(b_np).float().unsqueeze(0)  # [1, T_b, 1024]

    T_a = a.shape[1]
    T_b = b.shape[1]
    T_m = int((T_a + T_b) / 2)

    # Interpolate to T_m
    a_interp = F.interpolate(a.transpose(1, 2), size=T_m, mode="linear", align_corners=True).transpose(1, 2)
    b_interp = F.interpolate(b.transpose(1, 2), size=T_m, mode="linear", align_corners=True).transpose(1, 2)

    # Create weights from 0 to 1 across T_m and blend
    weights = torch.linspace(0, 1, T_m).view(1, T_m, 1)
    blended = (1 - weights) * a_interp + weights * b_interp  # [1, T_m, 1024]

    return blended


###############################################################################
# Entry point
###############################################################################
def main():
    # Initialization of the argument parser and the logger
    arg_parser = define_argument_parser()
    args = arg_parser.parse_args()
    logger = configure_logger(args)

    # Download the required model
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_path, config_path, vocoder_path, vocoder_config_path, _ = download_model_by_name(model_name)

    # load fine tuned model on VCTK corpus, can be skipped for using the published model

    model_dir = pathlib.Path(args.model_dir)
    config = load_config(model_dir / "config.json")
    tts_model = setup_tts_model(config)
    tts_model.load_checkpoint(config, checkpoint_path=model_dir / "best_model.pth", checkpoint_dir=model_dir, eval=True)
    if torch.cuda.is_available():
        tts_model.cuda

    # TODO SLM: this part is hardcoded
    # create Palette synthesis
    refwav = [model_dir / "ref_samples/p226_023.wav", model_dir / "ref_samples/p262_023.wav"]
    refwav = [
        "models/GPT_XTTS_v2.0_vctk_rfrm/ref_samples/p226_023.wav",
        "models/GPT_XTTS_v2.0_vctk_rfrm/ref_samples/p262_023.wav",
    ]
    wghts = [[1.25, -0.25], [1.0, 0.0]]
    assert len(wghts) <= 2

    # Ensure directory is available
    output_file = pathlib.Path(args.output_file)
    if not output_file.parent.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)

    # Synthesize everything
    audios = []
    for i in range(len(wghts)):
        audio = tts_model.synthesize(
            text=args.text,
            config=config,
            speaker_id=None,
            voice_dirs=None,
            d_vector=None,
            temperature=0.9,
            # speed=1.1,
            repetition_penalty=20.0,
            speaker_wav=refwav,
            audio_weights=wghts[i],  # TODO: SLM hardcoded for the moment
            language="en",
        )
        audios.append(audio)

        if args.dump_individual:
            audio_out = np.asarray(audio["wav"])
            cur_output_file = output_file.parent / f"{output_file.stem}_weights-{'_'.join([str(x) for x in wghts[i]])}.wav"
            logger.debug(f"Individual dump of {cur_output_file}")
            wave_write(cur_output_file, config.audio.output_sample_rate, audio_out)



    if len(audios) > 1:
        logger.info("There are multiple weights combination provided, use OMBRE")
        audio = tts_model.synthesize(
            text=args.text,
            config=config,
            speaker_id=None,
            voice_dirs=None,
            d_vector=None,
            temperature=0.9,
            # speed=1.1,
            repetition_penalty=20.0,
            speaker_wav=refwav,
            audio_weights=[0.5, 0.5],  # NOTE SLM: equal contribution of both references
            language="en",
        )
        audios.append(audio)

        if args.dump_individual:
            audio_out = np.asarray(audio["wav"])
            cur_output_file = output_file.parent / f"{output_file.stem}_weights-{'_'.join([str(x) for x in [[0.5, 0.5]]])}.wav"
            logger.debug(f"Individual dump of {cur_output_file}")
            wave_write(cur_output_file, config.audio.output_sample_rate, audio_out)

        # create ombré audio
        with torch.no_grad():
            if args.inverse:
                logger.info("Inverse OMBRE is used")
                ombre_latent = blend_latents_numpy(audios[0]["gpt_latents"][0], audios[1]["gpt_latents"][0])
            else:
                ombre_latent = blend_latents_numpy(audios[1]["gpt_latents"][0], audios[0]["gpt_latents"][0])
            audio_out = tts_model.hifigan_decoder(ombre_latent, g=audios[2]["speaker_embedding"]).cpu().squeeze()
            audio_out = np.asarray(audio_out)
    else:
        logger.info("There one weight combination provided, use PALETTE")
        audio = audios[0]
        audio_out = np.asarray(audio["wav"])

    wave_write(output_file, config.audio.output_sample_rate, audio_out)


###############################################################################
# Wrapping for directly calling the scripts
###############################################################################
if __name__ == "__main__":
    main()
