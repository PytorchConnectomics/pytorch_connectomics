from torch_connectomics.model.loss import *
from torch_connectomics.libs.vis import visualize, visualize_aff
from torch_connectomics.io import *
import numpy as np

def train(args, train_loader, model, criterion, 
          optimizer, scheduler, logger, writer, regularization=None, pre_iter=0):
    record = AverageMeter()
    model.train()

    optimizer.zero_grad()

    for iteration, batch in enumerate(train_loader):
        iter_total = pre_iter + iteration
        if args.task == 22:
            _, volume, seg_mask, class_weight, _, label, out_skeleton = batch
        else:
            if args.loss_weight_opt == 0:
                _, volume, label, class_weight = batch
            elif args.loss_weight_opt in [1,2]:
                _, volume, label, class_weight, extra = batch
                extra_label, extra_weight = extra

        volume, label = volume.to(args.device), label.to(args.device)
        class_weight = class_weight.to(args.device)
        if args.loss_weight_opt in [1,2]:
            extra_label, extra_weight = extra_label.to(args.device), extra_weight.to(args.device)

        output = model(volume)
       
        #visualize_aff(volume, label, output, iter_total, writer)
        # overall loss
        if args.loss_weight_opt == 0:
            loss = criterion(output, label, class_weight)
        else:
            loss = criterion(output[:,:label.shape[1]], label, class_weight)

        if args.loss_weight_opt in [1,2]:
            loss += criterion(output[:, label.shape[1]:], extra_label, extra_weight)

        if regularization is not None:
            loss += regularization(output)
        record.update(loss, args.batch_size) 

        # compute gradient and do Adam step
        loss.backward()
        if (iteration+1) % args.iteration_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (iteration+1) % (10*args.iteration_step) == 0:
            print('[Iteration %d] train_loss=%0.4f lr=%.5f' % (iter_total, \
                  record.avg, optimizer.param_groups[0]['lr']))
            if logger is not None:
                logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iter_total, \
                        loss.item(), optimizer.param_groups[0]['lr']))
                logger.flush()
            if writer is not None:
                writer.add_scalar('Loss', record.avg, iter_total)
                if (iteration+1) % (50*args.iteration_step) == 0:
                    if args.task == 0:
                        visualize_aff(volume, label, output, iter_total, writer)
                    elif args.task == 1 or args.task == 2 or args.task == 22:
                        visualize(volume, label, output, iter_total, writer)
                    elif args.task == 11:
                        visualize(volume, label, output, iter_total, writer, composite=True)
                    #print('weight factor: ', weight_factor) # debug
            scheduler.step(record.avg)
            record.reset()

        #Save model
        if (iter_total+1) % args.iteration_save == 0:
            torch.save(model.state_dict(), args.output+('/volume_%d%s.pth' % (iter_total, args.finetune)))

        # Terminate
        if iteration >= args.iteration_total:
            break    #     
