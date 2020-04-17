import torch
import numpy as np

def train(args, train_loader, model, criterion, 
          optimizer, scheduler, monitor, pre_iter=0):
    # setup
    model.train()
    monitor.reset()
    optimizer.zero_grad()

    loss = 0
    for iteration, batch in enumerate(train_loader):
        iter_total = pre_iter+iteration

        # load data
        _, volume, target, weight = batch

        # prediction
        volume = torch.from_numpy(volume).to(args.device, dtype=torch.float)
        pred = model(volume)
        #print(volume.size(), output.size())
       
        loss += criterion.eval(pred, target, weight)

        # compute gradient
        if (iteration+1) % args.iteration_step == 0:
            optimizer.zero_grad()
            loss.backward()
            loss = 0
            optimizer.step()

        # logging and update record
        do_vis = monitor.update(scheduler, iter_total, loss, optimizer.param_groups[0]['lr']) 
        if do_vis:
            monitor.visualize(volume, torch.from_numpy(target[0]), pred, iter_total)
        #Save model
        if (iter_total+1) % args.iteration_save == 0:
            torch.save(model.state_dict(), args.output_path+('/volume_%d%s.pth' % (iter_total, args.finetune)))

        # Terminate
        if iteration >= args.iteration_total:
            break
