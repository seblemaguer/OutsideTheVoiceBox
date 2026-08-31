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

# Configuration
import hydra
from omegaconf import DictConfig
from outside_voice_box.config import convert_paths, str_config


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

# # Add performative options
# parser.add_argument(
#     "-D",
#     "--dump_individual",
#     action="store_true",
#     help="Dump each intermediate reference weighted file (in the same directory than the final output file)",
# )
# parser.add_argument("-I", "--inverse", action="store_true", help="Inverse ombre blending")

# # Add arguments
# parser.add_argument("model_dir", help="The directory of themodel")
# parser.add_argument("text", help="The text to synthesize")
# parser.add_argument("output_file", help="The output wav file")


@hydra.main(config_path=str(pathlib.Path(__file__).parent), config_name="default_config", version_base=None)
def main(config: DictConfig):

    # Adapt the configuration to be ready to be used
    convert_paths(config)
    config.log_level = eval(config.log_level)

    # Configure the logger
    logging_config = dict(
        version=1,
        disable_existing_logger=True,
        formatters={
            "f": {
                # "format": "[%(asctime)s] [%(levelname)s] — [%(name)s — %(funcName)s:%(lineno)d] %(message)s",
                "format": "[%(levelname)s] — [%(name)s] %(message)s",
                "datefmt": "%d/%b/%Y: %H:%M:%S ",
            }
        },
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": config.log_level,
            }
        },
        root={"handlers": ["h"], "level": config.log_level},
    )
    dictConfig(logging_config)
    logging.basicConfig(level=config.log_level)

    logger = logging.getLogger("xtts")

    logger.info(str_config(config))

    # Download the required model
    model_name = config.model.name
    model_path, config_path, vocoder_path, vocoder_config_path, _ = download_model_by_name(model_name)

    # load fine tuned model on VCTK corpus, can be skipped for using the published model
    model_config = load_config(config.model.config_file)
    tts_model = setup_tts_model(model_config)
    tts_model.load_checkpoint(model_config, checkpoint_path=config.model.checkpoint_file, checkpoint_dir=config.model.checkpoint_file.parent, eval=True)
    if torch.cuda.is_available():
        tts_model.cuda

    # create Palette synthesis
    refwav = [str(p.resolve()) for p in config.reference_wav_files]
    wghts = config.parameters.weights
    assert len(wghts) <= 2

    # Ensure directory is available
    output_file = config.data.output_file
    if not output_file.parent.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)

    # Synthesize everything
    audios = []
    for i in range(len(wghts)):
        audio = tts_model.synthesize(
            text=config.data.text,
            config=model_config,
            speaker_id=None,
            voice_dirs=None,
            d_vector=None,
            temperature=config.parameters.temperature,
            # speed=config.parameters.speed,
            repetition_penalty=config.parameters.repetition_penalty,
            speaker_wav=refwav,
            audio_weights=wghts[i],
            language=config.parameters.language,
        )
        audios.append(audio)

        # TODO: integrate that again
        # if args.dump_individual:
        #     audio_out = np.asarray(audio["wav"])
        #     cur_output_file = output_file.parent / f"{output_file.stem}_weights-{'_'.join([str(x) for x in wghts[i]])}.wav"
        #     logger.debug(f"Individual dump of {cur_output_file}")
        #     wave_write(cur_output_file, model_config.audio.output_sample_rate, audio_out)



    if len(audios) > 1:
        logger.info("There are multiple weights combination provided, use OMBRE")
        audio = tts_model.synthesize(
            text=config.data.text,
            config=model_config,
            speaker_id=None,
            voice_dirs=None,
            d_vector=None,
            temperature=config.parameters.temperature,
            # speed=1.1,
            repetition_penalty=config.parameters.repetition_penalty,
            speaker_wav=refwav,
            audio_weights=[0.5, 0.5],  # NOTE SLM: equal contribution of both references
            language=config.parameters.language,
        )
        audios.append(audio)

        # if args.dump_individual:
        #     audio_out = np.asarray(audio["wav"])
        #     cur_output_file = output_file.parent / f"{output_file.stem}_weights-{'_'.join([str(x) for x in [[0.5, 0.5]]])}.wav"
        #     logger.debug(f"Individual dump of {cur_output_file}")
        #     wave_write(cur_output_file, model_config.audio.output_sample_rate, audio_out)

        # create ombré audio
        with torch.no_grad():
            if config.parameters.inverse:
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

    wave_write(output_file, model_config.audio.output_sample_rate, audio_out)


###############################################################################
# Wrapping for directly calling the scripts
###############################################################################
if __name__ == "__main__":
    main()
