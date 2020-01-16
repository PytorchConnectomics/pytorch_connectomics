from torch_connectomics.model.loss import *
from torch_connectomics.libs.vis import visualize, visualize_aff
from torch_connectomics.io import *
import numpy as np

def train(args, train_loader, model, criterion, 
          optimizer, scheduler, logger, writer, regularization=None):
    record = AverageMeter()
    model.train()

    # for iteration, (_, volume, label, class_weight, _) in enumerate(train_loader):
    optimizer.zero_grad()

    for iteration, batch in enumerate(train_loader):
        if args.task == 22:
            _, volume, seg_mask, class_weight, _, label, out_skeleton = batch
        else:
            _, volume, label, class_weight, _ = batch

        volume, label = volume.to(args.device), label.to(args.device)
        class_weight = class_weight.to(args.device)
        output = model(volume)

        if regularization is not None:
            loss = criterion(output, label, class_weight) + regularization(output)
        else:
            loss = criterion(output, label, class_weight)
        record.update(loss, args.batch_size) 

        # compute gradient and do Adam step
        loss.backward()
        if (iteration+1) % args.iteration_step == 0:
            optimizer.step()
            optimizer.zero_grad()


        if (iteration+1) % (10*args.iteration_step) == 0:
            print('[Iteration %d] train_loss=%0.4f lr=%.5f' % (iteration, \
                  record.avg, optimizer.param_groups[0]['lr']))

            writer.add_scalar('Loss', record.avg, iteration)

            # if tensorboard not available
            #logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration, \
            #        loss.item(), optimizer.param_groups[0]['lr']))
            #logger.flush()

            scheduler.step(record.avg)
            record.reset()
            
            if (iteration+1) % (50*args.iteration_step) == 0:
                if args.task == 0:
                    visualize_aff(volume, label, output, iteration, writer)
                elif args.task == 1 or args.task == 2 or args.task == 22:
                    visualize(volume, label, output, iteration, writer)
                elif args.task == 11:
                    visualize(volume, label, output, iteration, writer, composite=True)
                #print('weight factor: ', weight_factor) # debug
                # debug
                # if iteration < 50:
                #     fl = h5py.File('debug_%d_h5' % (iteration), 'w')
                #     output = label[0].cpu().detach().numpy().astype(np.uint8)
                #     print(output.shape)
                #     fl.create_dataset('main', data=output)
                #     fl.close()

        #Save model
        if (iteration+1) % args.iteration_save == 0 or iteration >= args.iteration_total:
            torch.save(model.state_dict(), args.output+('/volume_%d%s.pth' % (iteration, args.finetune)))

        # Terminate
        if iteration >= args.iteration_total:
            break    #     
