import torch
import numpy as np

def train(args, train_loader, model, criterion, 
          optimizer, scheduler, monitor, pre_iter=0):
    # setup
    model.train()
    monitor.reset()
    optimizer.zero_grad()

    for iteration, batch in enumerate(train_loader):
        iter_total = pre_iter+iteration

        # load data
        _, volume, label, mask = batch

        # prediction
        volume = torch.from_numpy(volume).to(args.device)
        pred = model(volume)
        #print(volume.size(), output.size())
       
        loss = criterion.eval(label, pred, mask)

        # compute gradient
        loss.backward()
        if (iteration+1) % args.iteration_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        # logging and update record
        do_vis = monitor.update(scheduler, iter_total, loss, optimizer.param_groups[0]['lr']) 
        if do_vis:
            monitor.vis(volume, torch.from_numpy(label), output, iter_total)
        #Save model
        if (iter_total+1) % args.iteration_save == 0:
            torch.save(model.state_dict(), args.output+('/volume_%d%s.pth' % (iter_total, args.finetune)))

        # Terminate
        if iteration >= args.iteration_total:
            break
