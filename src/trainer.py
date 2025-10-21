import os
import time
import datetime
import matplotlib.pyplot as plt
# import matplotlib 
# import matplotlib.cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.image as mpimg
import torch
import numpy as np
from tqdm import tqdm
import wandb
import time
from src.data_loader import get_dataloader
from src.metrics import compute_metric_mask
from src.utils import add_dict_prefix, formatted_time, print_model_summary, find_trainable_layers
from src.models.model import GDTS   

class trainer(object):
    def __init__(self, args):
        self.args = args
        # initialize data loaders
        self.data_loaders = dict()
        self.data_loaders['train'] = get_dataloader(args, set_name='train')
        self.data_loaders['valid'] = get_dataloader(args, set_name='valid')
        # initialize device
        self.device = self._set_device()
        # initialize network
        self.net = GDTS(self.args, self.device).to(self.device)

        # Prepare log curve file and initialize best validation metrics
        self.log_curve_file = os.path.join(self.args.model_dir, 'log_curve.txt')

        # Best metrics
        self.best_metrics = self.net.init_best_metrics()
        self.best_metrics_epochs = {k: -1 for k in self.best_metrics.keys()}

    def _save_checkpoint(self, epoch, best_epoch=False):
        """
        Save model and optimizer states
        """
        saved_models_path = os.path.join(self.args.model_dir, 'saved_models')
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)
        # Save current checkpoint
        if not best_epoch:
            saved_model_name = os.path.join(saved_models_path, 'epoch_' +str(epoch).zfill(3) + '.pt')
        else:  # best model name
            saved_model_name = os.path.join(
                saved_models_path, 'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
        }, saved_model_name)

    def _load_checkpoint(self, load_checkpoint):
        """
        Load a pre-trained model. Can then be used to test or resume training.
        """
        if load_checkpoint is not None:
            # Create load model path
            if load_checkpoint == 'best':
                saved_model_name = os.path.join(self.args.model_dir, 'saved_models', 'best_model.pt')
            else:  # Load specific checkpoint
                assert int(load_checkpoint) > 0, \
                    "Check args.load_model. Must be an integer > 0"
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models', 'epoch_' + str(load_checkpoint).zfill(3) + '.pt')
            print("\nSaved model path:", saved_model_name)
            # Load model
            if os.path.isfile(saved_model_name):
                print('Loading checkpoint ...')
                checkpoint = torch.load(saved_model_name,
                                        map_location=self.device)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(
                    checkpoint['model_state_dict'])
                print('Loaded checkpoint at epoch', model_epoch, '\n')
                return model_epoch
            else:
                raise ValueError("No such pre-trained model:", saved_model_name)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')

    def _load_or_restart(self):
        """
        Load a pre-trained model to resume training or restart from scratch.
        Can start from scratch or resume training, depending on input
        self.args.load_checkpoint parameter.
        """
        # load pre-trained model to resume training
        if self.args.load_checkpoint is not None:
            loaded_epoch = self._load_checkpoint(self.args.load_checkpoint)
            # start from the following epoch
            start_epoch = int(loaded_epoch) + 1
        else:
            start_epoch = 1
            # log_file header only the first time
            with open(self.log_curve_file, 'w') as f:
                f.write("epoch,learning_rate,"
                        "valid_ADE,valid_FDE,"
                        "valid_ADE_traj,valid_FDE_traj," +
                        ",".join(sorted(self.net.init_losses().keys())) +
                        "\n")
        return start_epoch

    def test(self, load_checkpoint):
        """
        Load a trained model and test it on the test set.
        """
        print('*** Test phase started ***')
        # some models do not need to be trained nor loaded
        if self.args.dataset == 'eth5':
            if self.net.is_trainable:
                best_epoch = self._load_checkpoint(load_checkpoint)
            else:
                best_epoch = load_checkpoint
        else:
            best_epoch = 1
        print('Testing ...')
        total_results = []
        for run_idx in range(self.args.num_test_runs):
            print(f"\nTest run #{run_idx} ...")
            test_metrics = self._evaluate_epoch(best_epoch, mode='valid')
            run_results = dict(**test_metrics)
            # print losses and metrics for run i
            print(f'Test_set: {self.args.test_set},',
                  f'test_run_idx: {run_idx},',
                  f'epoch: {load_checkpoint},',
                  ', '.join([f"{k}={v:.5f}" for k, v in run_results.items()]))
            total_results.append(run_results)
        average_results = {k: np.mean([i[k] for i in total_results])
                           for k in run_results}
        # print average losses and metrics for
        print("\n" + "#"*25)
        print("#"*5 + " FINAL RESULTS " + "#"*5)
        print("#" * 25)
        print(f'Test_set: {self.args.test_set},',
              f'epoch: {load_checkpoint},',
              ', '.join([f"{k}={v:.5f}" for k, v in average_results.items()]))

    def train(self):
        """
        Train the model. Wrapper for train_loop.
        """
        # find where to start
        start_epoch = self._load_or_restart()

        # print model info
        print_model_summary(self.net)

        # parameters to update
        params = find_trainable_layers(self.net)

        # Set optimizer
        self.optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)
        # Set scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

        # start training
        self._train_loop(start_epoch=start_epoch, end_epoch=self.args.num_epochs)

    def train_test(self):
        """
        Perform training and then test on the best validation epoch.
        """
        self.train()
        print()
        self.test(load_checkpoint='best')

    def _train_loop(self, start_epoch, end_epoch):
        """
        Train the model. Loop over the epochs, train and update network
        parameters, save model, check results on validation set,
        print results and save log data.
        """
        # saved metrics before validation begins
        valid_metrics = {"valid_ADE": 0, "valid_FDE": 0, "valid_ADE_traj": 0, "valid_FDE_traj": 0}

        # initial learning rate
        if self.scheduler is not None:
            learning_rate = self.optimizer.param_groups[0]['lr']
        else:
            learning_rate = self.args.learning_rate

        phase_name = 'Train'
        best_metric_name = self.net.best_valid_metric()

        print(f'*** {phase_name} phase started ***')
        print(f"Starting epoch: {start_epoch}, final epoch: {end_epoch}")

        self.current_date = datetime.datetime.now().strftime("%m%d")
        self.current_time = datetime.datetime.now().strftime("%H%M")
        if wandb.run is None and self.args.use_wandb:
            wandb.init(settings=wandb.Settings(start_method="thread"),
                       project="GDTS", config=self.args, entity="iadc_erwinsun",
                       group=f"{self.args.dataset}",
                       job_type=f"{self.args.test_set}",
                       tags=None, name=f'{self.args.test_set}_{self.current_date}_{self.current_time}')

        for epoch in range(start_epoch, end_epoch + 1):
            start_time = time.time()  # time epoch
            train_losses = self._train_epoch(epoch)

            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)  # save model
                print(f"Saved checkpoint at epoch {epoch}")

            # validation
            train_rng_state = torch.get_rng_state()
            if epoch >= self.args.start_validation and epoch % self.args.validate_every == 0:
                valid_metrics = self._evaluate_epoch(epoch, mode='valid')

                # comment some of this print, if it is too long
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]))
                print(', '.join([f"{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 valid_metrics.items()]))

                # update best model and best metrics
                for k, v in self.best_metrics.items():
                    current_metric_loss = valid_metrics["valid_" + k]

                    if current_metric_loss < v:
                        self.best_metrics[k] = current_metric_loss
                        self.best_metrics_epochs[k] = epoch
                        # save best model on best metric
                        if k == best_metric_name:
                            self._save_checkpoint(epoch, best_epoch=True)
                            print(f"Saved best model at epoch {epoch}")

                print(', '.join([f"best_{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 self.best_metrics.items()]))
                print(', '.join([f"best_epoch_{metric_name}="
                                 f"{metric_epoch}" for
                                 metric_name, metric_epoch in
                                 self.best_metrics_epochs.items()]))
                
                # save metrics to log_curve.txt
                with open(self.log_curve_file, 'a') as f:
                    f.write(','.join(str(m) for m in [
                        epoch, learning_rate,
                        valid_metrics["valid_ADE"],
                        valid_metrics["valid_FDE"],
                        valid_metrics["valid_ADE_traj"],
                        valid_metrics["valid_FDE_traj"]] +
                        [train_losses[loss_name] for loss_name in sorted(
                            train_losses)]) + '\n')
            else:
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]))
            # restore train rng state
            torch.set_rng_state(train_rng_state)
            self.scheduler.step()
            learning_rate = self.optimizer.param_groups[0]['lr']

            # save metrics to WandB
            if self.args.use_wandb and epoch >= self.args.start_validation:
                wandb.log({'learning_rate': learning_rate}, step=epoch)
                wandb.log(train_losses, step=epoch)
                # validation
                if epoch >= self.args.start_validation and epoch % self.args.validate_every == 0:
                    wandb.log(valid_metrics, step=epoch)
                    # best metrics
                    for k, v in self.best_metrics.items():
                        wandb.run.summary["best_" + k] = v
                    for k, v in self.best_metrics_epochs.items():
                        wandb.run.summary["best_epoch_" + k] = v

    def _train_epoch(self, epoch):
        """
        Train one epoch of the model on the whole training set.
        """
        self.net.train()  # train mode

        # INIT LOSSES and METRICS
        losses_epoch = self.net.init_losses()
        losses_coeffs = self.net.set_losses_coeffs()

        # Progress bar
        train_bar = tqdm(self.data_loaders['train'], ascii=True, ncols=100,
                         desc=f'Epoch {epoch}. Train batches')
        num_train_batches = len(self.data_loaders['train'])


        for batch_data, batch_id in train_bar:
            
            inputs, seq_list = self.net.prepare_inputs(batch_data, batch_id)
            del batch_data

            self.optimizer.zero_grad()  # sets grads to zero
            losses = self.net.get_loss(inputs, seq_list) 
            loss = torch.zeros(1).to(self.device)
            for loss_name, loss_value in losses.items():
                loss += losses_coeffs[loss_name]*loss_value
                losses_epoch[loss_name] += losses_coeffs[loss_name]*loss_value.item() 

            # Update network weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

            del inputs, batch_id
            del seq_list, losses, loss

        # update losses
        for loss_name in losses_epoch.keys():
            # mean losses over batches
            losses_epoch[loss_name] = losses_epoch[loss_name] / num_train_batches
        losses_epoch = add_dict_prefix(losses_epoch, prefix='train')

        return losses_epoch

    @torch.no_grad()
    def _evaluate_epoch(self, epoch, mode='valid'):
        """
        Loop over the validation or test set once. Compute metrics and save
        output trajectories.
        """
        self.net.eval()  # evaluation mode

        # INIT LOSSES and METRICS
        metrics_epoch = self.net.init_test_metrics()

        if wandb.run is None and self.args.use_wandb:
            wandb.init(settings=wandb.Settings(start_method="thread"),
                       project="GDTS", config=self.args, entity="iadc_erwinsun",
                       group=f"{self.args.dataset}",
                       job_type=f"{self.args.test_set}",
                       tags=None, name=None)

        # Progress bar
        evaluate_bar = tqdm(self.data_loaders[mode], ascii=True, ncols=100,
                            desc=f'Epoch {epoch}. {mode.title()} batches')


        # loop over batches
        total_time = 0
        num_input = 0
        for batch_data, batch_id in evaluate_bar:

            inputs, seq_list = self.net.prepare_inputs(batch_data, batch_id)
            del batch_data
            
            # compute metric_mask
            metric_mask = compute_metric_mask(seq_list)
            st = time.time()
            all_output, all_aux_outputs = self.net.forward(inputs, if_test=True) # (21,Tp+Tf,B,2) 
            total_time = total_time + time.time() - st
            num_input = num_input + inputs["abs_pixel_coord"].shape[1]

            # update metrics
            for metric_name in metrics_epoch.keys():
                # print('metric_name: ', metric_name)
                metrics_epoch[metric_name].extend(
                    self.net.compute_model_metrics(
                        metric_name=metric_name,
                        predictions=all_output,
                        # predictions=traj_init_guess,
                        metric_mask=metric_mask,
                        all_aux_outputs=all_aux_outputs,
                        inputs=inputs,
                        obs_length=self.args.obs_length,
                    ))

            del inputs, batch_id, all_output, all_aux_outputs
            del seq_list, metric_mask


        # update metrics
        evaluate_metrics = {}
        for metric_name in metrics_epoch.keys():
            array_metric = np.array(metrics_epoch[metric_name])
            evaluate_metrics[metric_name] = array_metric.mean()
            if "goal" in metric_name:
                evaluate_metrics[metric_name + '_std'] = array_metric.std()
        evaluate_metrics = add_dict_prefix(evaluate_metrics, prefix=mode)
        # print('total_time (ms): ', total_time*1000)
        # print('num_input: ', num_input)
        print('average_time for 1 input(ms/input): ', total_time*1000/num_input)
        return evaluate_metrics

    def _set_device(self):
        """
        Set the device for the experiment. GPU if available, else CPU.
        """
        # torch.cuda.is_available() already checked in parsed args
        device = torch.device(self.args.device)
        print('\nUsing device:', device)

        # Additional info when using cuda
        if device.type == 'cuda':
            print('Number of available GPUs:', torch.cuda.device_count())
            print('GPU name:', torch.cuda.get_device_name(0))
            print('Cuda version:', torch.version.cuda)
        print()
        return device
