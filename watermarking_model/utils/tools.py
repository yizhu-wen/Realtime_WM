import os
import datetime
import torch



def save_op(path, epoch, encoder, decoder, en_de_op):
    if not os.path.exists(path): os.makedirs(path)
    now = datetime.datetime.now()
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "en_de_op": en_de_op.state_dict(),
        },
        os.path.join(path,"none-"+ encoder.name + "_ep_{}_{}.pth.tar".format(epoch, now.strftime("%Y-%m-%d_%H_%M_%S"))),
    )
