import os
import torch
import yaml
import logging
import argparse
import wandb
import numpy as np
from torch.optim import Adam
from rich.progress import track
from torch.utils.data import DataLoader
from model.loss import Loss_identity
from utils.tools import save_op
from utils.optimizer import my_step
from itertools import chain
from torch.nn.functional import mse_loss
from dataset.data import collate_fn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from model.conv2_mel_modules import (
    Encoder,
    Decoder,
    Discriminator,
)
from dataset.data import WavDataset as MyDataset
import warnings
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if torch.cuda.is_available():
    pc = torch.backends.cuda.cufft_plan_cache
    pc.max_size = 512  # start here

    # optional warmup logic
    # run a few warmup batches first, then:
    used = pc.size
    headroom = int(pc.max_size * 1.2)  # 20 percent headroom
    pc.max_size = max(pc.max_size, headroom)
    print({"plans_in_use": used, "max_allowed": pc.max_size})




# Set random seed for reproducibility
def set_random_seed(seed: int):
    random.seed(seed)  # For Python random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # For PyTorch (GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimization for reproducibility


def generate_random_msg(batch_size, msg_length, device):
    # random [0, 1], mapped to [-1, 1]
    return (
        torch.randint(0, 2, (batch_size, 1, msg_length), device=device).float() * 2
    ) - 1


# Set random seed for reproducibility
SEED = 2022
set_random_seed(SEED)

logging_mark = "#" * 20
# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_wandb_key():
    """Return the wandb API key, or None to fall back to a cached `wandb login`.

    Checked before the dataset scan so a missing credential fails in seconds
    rather than after several minutes of scanning the corpus.
    """
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    if os.path.exists(os.path.expanduser("~/.netrc")):
        return None  # let wandb pick up the cached login
    raise SystemExit(
        "wandb.enabled is true in the train config but WANDB_API_KEY is not set.\n"
        "  Set it:  export WANDB_API_KEY=<your key>\n"
        "           (add that line to ~/.bashrc to make it permanent)\n"
        "  Or:      set wandb.enabled: false in the train config to train without logging."
    )


def main(configs):
    logging.info("main function")
    process_config, model_config, train_config = configs

    # Fail fast on a missing wandb credential, before the slow dataset scan.
    wandb_key = resolve_wandb_key() if train_config["wandb"]["enabled"] else None

    pre_step = 0
    # ---------------- get train dataset
    train_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="train"
    )
    val_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="val"
    )
    dev_audios = MyDataset(
        process_config=process_config, train_config=train_config, flag="test"
    )

    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size <= len(train_audios)
    train_audios_loader = DataLoader(
        train_audios,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,  # Enable pin memory for faster data transfer to GPU
        num_workers=20,  # Enable multiple workers for data loading
        persistent_workers=True,  # Keep workers alive between epochs
    )
    val_audios_loader = DataLoader(
        val_audios,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=20,
        persistent_workers=True,
    )
    dev_audios_loader = DataLoader(
        dev_audios,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=20,
        persistent_workers=True,
    )

    # Initialize wandb only if enabled in config
    if train_config["wandb"]["enabled"]:
        # Key comes from WANDB_API_KEY, or None to use a cached `wandb login`.
        # Never read a key committed to the config file.
        wandb.login(key=wandb_key)
        wandb.init(
            project=train_config["wandb"]["project"],
            name=train_config["wandb"]["name"],
            config={
                "dataset": "LibriSpeech",
                "data_size": "whole size divided by {}".format(
                    train_config["iter"]["data_divider"]
                ),
                "batch_size": train_config["optimize"]["batch_size"],
                "learning_rate": train_config["optimize"]["lr"],
                "delay_amt_second": train_config["watermark"]["delay_amt_second"],
                "future_amt_second": train_config["watermark"]["future_amt_second"],
                "nbits": train_config["watermark"]["length"],
                "epochs": train_config["iter"]["epoch"],
                "save_circle": train_config["iter"]["save_circle"],
                "show_circle": train_config["iter"]["show_circle"],
            },
        )
        test_loss_summary_table = wandb.Table(
            columns=[
                "test_loss",
                "test_wav_loss",
                "test_msg_loss",
                "test_loudness_loss",
                "test_acc_1",
                "test_acc_2",
                "test_avg_snr",
                "test_d_loss_on_encoded",
                "test_d_loss_on_cover",
            ]
        )
        test_audio_table = wandb.Table(
            columns=[
                "Original Audio",
                "Watermarked Audio",
                "Watermark Audio",
                "Original Amplitude Spectrogram",
                "Watermarked Amplitude Spectrogram",
                "Watermark Amplitude Spectrogram",
            ]
        )
    else:
        test_loss_summary_table = None

    # ---------------- build model
    msg_length = train_config["watermark"]["length"]

    encoder = Encoder(process_config, model_config, train_config, msg_length).to(device)
    decoder = Decoder(process_config, model_config, train_config, msg_length).to(device)
    # adv
    if train_config["adv"]:
        discriminator = Discriminator(process_config).to(device)
        d_op = Adam(
            params=discriminator.parameters(),  # No need for chain() with single model
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr=train_config["optimize"]["lr"],
        )
        lr_sched_d = StepLR(
            d_op,
            step_size=train_config["optimize"]["step_size"],
            gamma=train_config["optimize"]["gamma"],
        )
    # ---------------- optimizer
    en_de_op = Adam(
        params=chain(decoder.parameters(), encoder.parameters()),
        betas=train_config["optimize"]["betas"],
        eps=train_config["optimize"]["eps"],
        weight_decay=train_config["optimize"]["weight_decay"],
        lr=train_config["optimize"]["lr"],
    )
    lr_sched = StepLR(
        en_de_op,
        step_size=train_config["optimize"]["step_size"],
        gamma=train_config["optimize"]["gamma"],
    )

    # ---------------- Loss
    loss = Loss_identity()

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # ---------------- train
    logging.info(logging_mark + "\t" + "Begin Training" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_b = train_config["optimize"]["lambda_b"]
    hop_length = process_config["mel"]["hop_length"]
    offset_samples = 40480  # ((204+50)-1)*160
    global_step = 0
    train_len = len(train_audios_loader)

    for ep in range(1, epoch_num + 1):
        encoder.train()
        decoder.train()
        if train_config["adv"]:
            discriminator.train()
        step = 0
        logging.info("Epoch {}/{}".format(ep, epoch_num))
        train_avg_acc = [0, 0]
        train_avg_snr = 0
        train_avg_wav_loss = 0
        train_avg_msg_loss = 0
        train_avg_loudness_loss = 0
        train_avg_d_loss_on_encoded = 0
        train_avg_d_loss_on_cover = 0
        for sample in track(train_audios_loader):
            # inside the loop, e.g. every 500 steps
            global_step += 1
            step += 1
            b = sample["matrix"].shape[0]
            # ---------------- build watermark
            wav_matrix = sample["matrix"].to(device)
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, zeros_right = encoder(wav_matrix, msg, global_step)
            waveform_length = (zeros_right - 1) * hop_length if zeros_right > 1 else 0
            end = None if waveform_length == 0 else -waveform_length
            y_wm = wav_matrix + watermark
            decoded = decoder(y_wm, global_step)
            losses = loss.en_de_loss(
                wav_matrix[:, offset_samples:end],
                y_wm[:, offset_samples:end],
                msg,
                decoded,
            )
            # lamda_e = 1.
            # lamda_m = 10
            # lamda_b = 1
            if global_step < pre_step:
                sum_loss = lambda_m * losses[1]
            else:
                sum_loss = (
                    lambda_e * losses[0] + lambda_m * losses[1] + lambda_b * losses[2]
                )

            # adv
            if train_config["adv"]:
                lambda_a = train_config["optimize"]["lambda_a"]
                g_target_label_encoded = torch.full((b, 1), 1, device=device).float()
                d_on_encoded_for_enc = discriminator(y_wm[:, offset_samples:end])
                # target label for encoded images should be 'cover', because we want to fool the discriminator
                g_loss_adv = F.binary_cross_entropy_with_logits(
                    d_on_encoded_for_enc, g_target_label_encoded
                )
                sum_loss += lambda_a * g_loss_adv

            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                chain(encoder.parameters(), decoder.parameters()), max_norm=1.0
            )

            my_step(en_de_op, lr_sched, global_step, train_len)

            if train_config["adv"]:
                # sum_loss.backward() above also deposited the generator's
                # adversarial gradient on the discriminator (g_loss_adv flows
                # through it). Clear it so the discriminator is updated only by
                # its own objective, not by the term that tries to fool it.
                d_op.zero_grad()

                d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix[:, offset_samples:end])
                d_loss_on_cover = F.binary_cross_entropy_with_logits(
                    d_on_cover, d_target_label_cover
                )
                d_loss_on_cover.backward()

                d_target_label_encoded = torch.full((b, 1), 0, device=device).float()
                d_on_encoded = discriminator(y_wm[:, offset_samples:end].detach())
                # target label for encoded images should be 'encoded', because we want discriminator fight with encoder
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                    d_on_encoded, d_target_label_encoded
                )
                d_loss_on_encoded.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                my_step(d_op, lr_sched_d, global_step, train_len)

            decoder_acc = [
                ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
            ]
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(
                mse_loss(wav_matrix.detach(), zero_tensor)
                / mse_loss(wav_matrix.detach(), y_wm.detach())
            )
            norm2 = mse_loss(wav_matrix.detach(), zero_tensor)

            # Update averages
            train_avg_acc[0] += decoder_acc[0]
            train_avg_acc[1] += decoder_acc[1]
            train_avg_snr += snr.item()
            train_avg_wav_loss += losses[0].item()
            train_avg_msg_loss += losses[1].item()
            train_avg_loudness_loss += losses[2].item()
            if train_config["adv"]:
                train_avg_d_loss_on_cover += d_loss_on_cover.item()
                train_avg_d_loss_on_encoded += d_loss_on_encoded.item()

            if step % show_circle == 0:
                logging.info("-" * 100)
                msg_line = (
                    "step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f}"
                    " - acc:[{:.8f},{:.8f}] - snr:{:.8f} - norm:{:.8f}"
                    " - patch_num:{} - pad_num:{} - wav_len:{}".format(
                        step,
                        losses[0],
                        losses[1],
                        losses[2],
                        decoder_acc[0],
                        decoder_acc[1],
                        snr,
                        norm2,
                        sample["patch_num"].tolist(),
                        sample["pad_num"].tolist(),
                        wav_matrix.shape[-1],
                    )
                )
                if train_config["adv"]:
                    msg_line += " - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                        d_loss_on_encoded.item(), d_loss_on_cover.item()
                    )
                logging.info(msg_line)

        train_avg_acc[0] /= step
        train_avg_acc[1] /= step
        train_avg_snr /= step
        train_avg_wav_loss /= step
        train_avg_msg_loss /= step
        train_avg_loudness_loss /= step
        train_avg_d_loss_on_encoded /= step
        train_avg_d_loss_on_cover /= step

        logging.info("#t" * 60)
        logging.info(
            "epoch:{} [train] - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f}"
            " - acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                ep,
                train_avg_wav_loss,
                train_avg_msg_loss,
                train_avg_loudness_loss,
                train_avg_acc[0],
                train_avg_acc[1],
                train_avg_snr,
                train_avg_d_loss_on_encoded,
                train_avg_d_loss_on_cover,
            )
        )

        train_metrics = {
            "train/wav_loss": train_avg_wav_loss,
            "train/msg_loss": train_avg_msg_loss,
            "train/loudness_loss": train_avg_loudness_loss,
            "train/acc_1": train_avg_acc[0],
            "train/acc_2": train_avg_acc[1],
            "train/snr": train_avg_snr,
            "train/d_loss_on_encoded": train_avg_d_loss_on_encoded,
            "train/d_loss_on_cover": train_avg_d_loss_on_cover,
        }

        # if ep % save_circle == 0 or ep == 1 or ep == 2:
        if ep % save_circle == 0:
            path = os.path.join(train_config["path"]["ckpt"], "pth")
            save_op(
                path,
                ep,
                encoder,
                decoder,
                en_de_op,
                discriminator if train_config["adv"] else None,
                d_op if train_config["adv"] else None,
            )

        # ---------------- validation
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            if train_config["adv"]:
                discriminator.eval()
            val_avg_acc = [0, 0]
            val_avg_snr = 0
            val_avg_wav_loss = 0
            val_avg_msg_loss = 0
            val_avg_loudness_loss = 0
            val_avg_d_loss_on_encoded = 0
            val_avg_d_loss_on_cover = 0
            count = 0
            for sample in track(val_audios_loader):
                count += 1
                b = sample["matrix"].shape[0]
                # ---------------- build watermark
                wav_matrix = sample["matrix"].to(device)
                msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
                watermark, zeros_right = encoder(wav_matrix, msg, global_step)
                waveform_length = (
                    (zeros_right - 1) * hop_length if zeros_right > 1 else 0
                )
                end = None if waveform_length == 0 else -waveform_length
                y_wm = wav_matrix + watermark
                decoded = decoder(y_wm, global_step)
                losses = loss.en_de_loss(
                    wav_matrix[:, offset_samples:end],
                    y_wm[:, offset_samples:end],
                    msg,
                    decoded,
                )
                # adv
                if train_config["adv"]:
                    lambda_a = train_config["optimize"]["lambda_a"]
                    g_target_label_encoded = torch.full(
                        (b, 1), 1, device=device
                    ).float()
                    d_on_encoded_for_enc = discriminator(y_wm[:, offset_samples:end])
                    g_loss_adv = F.binary_cross_entropy_with_logits(
                        d_on_encoded_for_enc, g_target_label_encoded
                    )
                if train_config["adv"]:
                    d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                    d_on_cover = discriminator(wav_matrix[:, offset_samples:end])
                    d_loss_on_cover = F.binary_cross_entropy_with_logits(
                        d_on_cover, d_target_label_cover
                    )
                    d_target_label_encoded = torch.full(
                        (b, 1), 0, device=device
                    ).float()
                    d_on_encoded = discriminator(y_wm[:, offset_samples:end].detach())
                    d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                        d_on_encoded, d_target_label_encoded
                    )

                decoder_acc = [
                    ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                    ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                ]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(
                    mse_loss(wav_matrix.detach(), zero_tensor)
                    / mse_loss(wav_matrix.detach(), y_wm.detach())
                )
                val_avg_acc[0] += decoder_acc[0]
                val_avg_acc[1] += decoder_acc[1]
                val_avg_snr += snr.item()
                val_avg_wav_loss += losses[0].item()
                val_avg_msg_loss += losses[1].item()
                val_avg_loudness_loss += losses[2].item()
                if train_config["adv"]:
                    val_avg_d_loss_on_cover += d_loss_on_cover.item()
                    val_avg_d_loss_on_encoded += d_loss_on_encoded.item()
            val_avg_acc[0] /= count
            val_avg_acc[1] /= count
            val_avg_snr /= count
            val_avg_wav_loss /= count
            val_avg_msg_loss /= count
            val_avg_loudness_loss /= count
            val_avg_d_loss_on_encoded /= count
            val_avg_d_loss_on_cover /= count
            logging.info("#e" * 60)
            logging.info(
                "epoch:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                    ep,
                    val_avg_wav_loss,
                    val_avg_msg_loss,
                    val_avg_loudness_loss,
                    val_avg_acc[0],
                    val_avg_acc[1],
                    val_avg_snr,
                    val_avg_d_loss_on_encoded,
                    val_avg_d_loss_on_cover,
                )
            )

        val_metrics = {
            "val/wav_loss": val_avg_wav_loss,
            "val/msg_loss": val_avg_msg_loss,
            "val/tfloudness_loss": val_avg_loudness_loss,
            "val/acc_1": val_avg_acc[0],
            "val/acc_2": val_avg_acc[1],
            "val/snr": val_avg_snr,
            "val/d_loss_on_encoded": val_avg_d_loss_on_encoded,
            "val/d_loss_on_cover": val_avg_d_loss_on_cover,
        }

        if train_config["wandb"]["enabled"]:
            wandb.log({**train_metrics, **val_metrics})

    with torch.inference_mode():
        encoder.eval()
        decoder.eval()
        if train_config["adv"]:
            discriminator.eval()
        test_avg_acc = [0, 0]
        test_avg_snr = 0
        test_avg_wav_loss = 0
        test_avg_msg_loss = 0
        test_avg_loudness_loss = 0
        test_avg_d_loss_on_encoded = 0
        test_avg_d_loss_on_cover = 0
        count = 0
        for sample in track(dev_audios_loader):
            count += 1
            b = sample["matrix"].shape[0]
            # ---------------- build watermark
            wav_matrix = sample["matrix"].to(device)
            msg = generate_random_msg(wav_matrix.size(0), msg_length, device)
            watermark, zeros_right = encoder(wav_matrix, msg, global_step)
            waveform_length = (zeros_right - 1) * hop_length if zeros_right > 1 else 0
            end = None if waveform_length == 0 else -waveform_length
            y_wm = wav_matrix + watermark
            decoded = decoder(y_wm, global_step)
            losses = loss.en_de_loss(
                wav_matrix[:, offset_samples:end],
                y_wm[:, offset_samples:end],
                msg,
                decoded,
            )
            # adv
            if train_config["adv"]:
                lambda_a = train_config["optimize"]["lambda_a"]
                g_target_label_encoded = torch.full((b, 1), 1, device=device).float()
                d_on_encoded_for_enc = discriminator(y_wm[:, offset_samples:end])
                g_loss_adv = F.binary_cross_entropy_with_logits(
                    d_on_encoded_for_enc, g_target_label_encoded
                )
            if train_config["adv"]:
                d_target_label_cover = torch.full((b, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix[:, offset_samples:end])
                d_loss_on_cover = F.binary_cross_entropy_with_logits(
                    d_on_cover, d_target_label_cover
                )

                d_target_label_encoded = torch.full((b, 1), 0, device=device).float()
                d_on_encoded = discriminator(y_wm[:, offset_samples:end].detach())
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(
                    d_on_encoded, d_target_label_encoded
                )

            decoder_acc = [
                ((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
                ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(),
            ]
            zero_tensor = torch.zeros(wav_matrix.shape).to(device)
            snr = 10 * torch.log10(
                mse_loss(wav_matrix.detach(), zero_tensor)
                / mse_loss(wav_matrix.detach(), y_wm.detach())
            )
            test_avg_acc[0] += decoder_acc[0]
            test_avg_acc[1] += decoder_acc[1]
            test_avg_snr += snr.item()
            test_avg_wav_loss += losses[0].item()
            test_avg_msg_loss += losses[1].item()
            test_avg_loudness_loss += losses[2].item()
            if train_config["adv"]:
                test_avg_d_loss_on_cover += d_loss_on_cover.item()
                test_avg_d_loss_on_encoded += d_loss_on_encoded.item()

        test_avg_acc[0] /= count
        test_avg_acc[1] /= count
        test_avg_snr /= count
        test_avg_wav_loss /= count
        test_avg_msg_loss /= count
        test_avg_loudness_loss /= count
        test_avg_d_loss_on_encoded /= count
        test_avg_d_loss_on_cover /= count
        test_loss = (
            test_avg_acc[0]
            + test_avg_acc[1]
            + test_avg_snr
            + test_avg_wav_loss
            + test_avg_msg_loss
            + test_avg_d_loss_on_encoded
            + test_avg_d_loss_on_cover
        )
        logging.info("#test" * 20)
        logging.info(
            "Test: wav_loss:{:.8f} - msg_loss:{:.8f} - tfloudness_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{:.8f} - d_loss_on_cover:{:.8f}".format(
                test_avg_wav_loss,
                test_avg_msg_loss,
                test_avg_loudness_loss,
                test_avg_acc[0],
                test_avg_acc[1],
                test_avg_snr,
                test_avg_d_loss_on_encoded,
                test_avg_d_loss_on_cover,
            )
        )

        if train_config["wandb"]["enabled"]:
            # Log the audio_table to wandb
            wandb.log({"test_audio_table": test_audio_table})

        if train_config["wandb"]["enabled"] and test_loss_summary_table is not None:
            test_loss_summary_table.add_data(
                test_loss,
                test_avg_wav_loss,
                test_avg_msg_loss,
                test_avg_loudness_loss,
                test_avg_acc[0],
                test_avg_acc[1],
                test_avg_snr,
                test_avg_d_loss_on_encoded,
                test_avg_d_loss_on_cover,
            )

            wandb.log(
                {
                    "test_loss_summary_table": test_loss_summary_table,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        help="path to process.yaml",
        default=r"./config/process.yaml",
    )
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        help="path to model.yaml",
        default=r"./config/model.yaml",
    )
    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        help="path to train.yaml",
        default=r"./config/train.yaml",
    )
    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(open(args.process_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(configs)
