import os
import datetime
import torch


def save_op(path, epoch, encoder, decoder, en_de_op, discriminator=None, d_op=None):
    os.makedirs(path, exist_ok=True)
    now = datetime.datetime.now()
    state = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "en_de_op": en_de_op.state_dict(),
    }
    # Without these an adversarially-trained run cannot be resumed: the
    # discriminator would restart from scratch against a trained encoder.
    if discriminator is not None:
        state["discriminator"] = discriminator.state_dict()
    if d_op is not None:
        state["d_op"] = d_op.state_dict()
    torch.save(
        state,
        os.path.join(
            path,
            "none-"
            + encoder.name
            + "_ep_{}_{}.pth.tar".format(epoch, now.strftime("%Y-%m-%d_%H_%M_%S")),
        ),
    )
