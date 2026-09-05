

def my_step(opt, lr_sched, cur_iter, train_len):
    opt.step()
    opt.zero_grad()
    if cur_iter % train_len == 0:
        lr_sched.step()
