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
            if args.loss_type in [1,4]:
                if args.loss_weight_opt == 0:
                    _, volume, label, class_weight = batch
                elif args.loss_weight_opt in [1,2,3]:
                    _, volume, label, extra = batch
                    class_weight, extra_label, extra_weight = extra
            else:
                pos, volume, label = batch

        volume = volume.to(args.device)
        output = model(volume)
        #print(volume.size(), output.size())
        
        loss = 0
        if args.loss_type in [0,1,4]:
            if args.loss_weight_opt in [1,2,3]:
                label = torch.cat((label,extra_label),1)
                class_weight = torch.cat((class_weight,extra_weight*args.loss_weight_val[args.loss_weight_opt]),1)
            label, class_weight = label.to(args.device), class_weight.to(args.device)
            loss += criterion[1][0]*criterion[0][0](output, label, class_weight)

        if args.loss_type in [2,3,4]:
            label = label.to(args.device)
            criterion_st = 0 if args.loss_type in [2,3] else 1
            for x in range(criterion_st,len(criterion[0])):
                loss += criterion[1][x]*criterion[0][x](output, label)
        # writeh5('pred.h5','main',(output.cpu().detach().numpy().squeeze()*255).astype(np.uint8))
        # writeh5('input.h5','main',(volume.cpu().detach().numpy().squeeze()*255).astype(np.uint8))
        # writeh5('gt.h5','main',(label.cpu().detach().numpy().squeeze()*255).astype(np.uint8))

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
