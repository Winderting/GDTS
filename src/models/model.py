import os
import torch
import torch.nn.functional as F
import numpy as np

from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.model_utils.U_net_CNN import UNet
from src.losses import Goal_BCE_loss
from src.metrics import ADE_best_of, FDE_best_of, compute_metric_mask
from src.models.model_utils.sampling_2D_map import TTST_test_time_sampling_trick
from src.models.diffusion import VarianceSchedule, TransformerConcatLinear, linear_beta_schedule
from src.models.model_utils.hist_traj_rnn_encoder import Encoding
from src.models.model_utils.model_registrar import ModelRegistrar
from src.models.common import derivative_of


class GDTS(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.is_trainable = True
        self.args = args
        self.device = device
        self.dataset = create_dataset(self.args.dataset)

        ##################
        # MODEL PARAMETERS
        ##################
        # Goal Module
        self.num_image_channels = 6
        self.enc_chs = (self.num_image_channels + self.args.obs_length, 32, 32, 64, 64, 64)
        self.dec_chs = (64, 64, 64, 32, 32)
        self.goal_module = UNet(enc_chs=self.enc_chs, dec_chs=self.dec_chs, out_chs=self.args.pred_length).to(self.device)

        # Augmented History Encoder
        self.registrar = ModelRegistrar(self.args.model_dir, "cuda")
        self.encoder = Encoding(self.registrar, "cuda", self.args.e_dim)

        # Diffusion Network
        self.var_sched = VarianceSchedule(num_steps=self.args.ddpm_step, beta_T=5e-2, mode='linear')
        self.diffnet = TransformerConcatLinear(context_dim=self.args.e_dim, tf_layer=2).to(self.device)

        # ddim
        self.betas = linear_beta_schedule(self.args.ddpm_step) #cosine_beta_schedule(self.args.ddpm_step)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)


        # saved_model_name = os.path.join(
        #             self.args.model_dir, 'saved_models',
        #             'goal_pretrain_best_model.pt')
        # if os.path.isfile(saved_model_name):
        #     print('Loading pretrained goal module...')
        #     checkpoint = torch.load(saved_model_name,
        #                             map_location=self.device)
        #     model_state_dict = checkpoint['model_state_dict']
        #     new_model_state_dict = {}
        #     for k in list(model_state_dict.keys()):
        #         new_k = k.replace('goal_module.', '')
        #         new_model_state_dict[new_k] = model_state_dict[k]
        #     self.goal_module.load_state_dict(new_model_state_dict)
        
    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float
        selected_inputs = {k: v.squeeze(0).float().to(self.device) if \
            torch.is_tensor(v) else v for k, v in batch_data.items()}
        # extract seq_list
        seq_list = selected_inputs["seq_list"]

        # state augmentation to [T, B, 8]: rel_px, rel_py, vx, vy, ax, ay, abs_px, abs_py
        num_agents = selected_inputs["abs_pixel_coord"].shape[1]
        abs_x = selected_inputs["abs_pixel_coord"]
        x_augmented = torch.zeros(self.args.seq_length, num_agents, 8, device=self.device)
        x_augmented[:,:,6:] = abs_x # original absolute px, py

        x = abs_x[:self.args.obs_length,:,:2] - abs_x[self.args.obs_length-1,:,:2] 
        vx = derivative_of(x, dt=1) 
        ax = derivative_of(vx, dt=1) 
        x_augmented[:self.args.obs_length,:,:2] = x
        x_augmented[:self.args.obs_length,:,2:4] = vx
        x_augmented[:self.args.obs_length,:,4:6] = ax

        y = abs_x[self.args.obs_length:,:,:2] - abs_x[self.args.obs_length-1,:,:2] 
        vy = derivative_of(y, dt=1) 
        ay = derivative_of(vy, dt=1) 
        x_augmented[self.args.obs_length:,:,:2] = y
        x_augmented[self.args.obs_length:,:,2:4] = vy
        x_augmented[self.args.obs_length:,:,4:6] = ay

        selected_inputs["x_augmented"] = x_augmented # TBC

        scene_name = batch_id["scene_name"][0]
        scene = self.dataset.scenes[scene_name]
        selected_inputs["scene"] = scene

        return selected_inputs, seq_list.detach()

    def init_losses(self):
        losses = {
            "diffusion_loss": 0,
            "goal_BCE_loss": 0,
        }
        return losses

    def set_losses_coeffs(self):

        losses_coeffs = {
            "diffusion_loss": 1,
            "goal_BCE_loss": 20,
        } 
        return losses_coeffs
    
    
    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            "FDE": [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "ADE": 1e9,
            "FDE": 1e9,
            "ADE_world": 1e9,
            "FDE_world": 1e9,
        }
        return best_metrics

    def best_valid_metric(self):
        return "ADE" # use ADE as the best metric

    def compute_loss_mask(self, seq_list, obs_length: int = 8):
        """
        Get a mask to denote whether to account predictions during loss
        computation. It is supposed to calculate losses for a person at
        time t only if his data exists from time 0 to time t.

        Parameters
        ----------
        seq_list : PyTorch tensor
            input is seq_list[1:]. Size = (seq_len,N_pedestrians). Boolean mask
            that is =1 if pedestrian i is present at time-step t.
        obs_length : int
            number of observation time-steps

        Returns
        -------
        loss_mask : PyTorch tensor
            Shape: (seq_len,N_pedestrians)
            loss_mask[t,i] = 1 if pedestrian i if present from beginning till time t
        """
        loss_mask = seq_list.cumprod(dim=0)
        # we should not compute losses for step 0, as ground_truth and
        # predictions are always equal there
        loss_mask[0:obs_length] = 1
        return loss_mask
    
    def compute_model_metrics(self,
                              metric_name,
                              predictions,
                              metric_mask,
                              all_aux_outputs,
                              inputs,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """

        # scale back to original dimension
        predictions = predictions.detach() * self.args.down_factor
        ground_truth = inputs["x_augmented"][:,:,6:8].detach()
        ground_truth = ground_truth.detach() * self.args.down_factor

        # convert to world coordinates
        scene = inputs["scene"]
        pred_world = []
        for i in range(predictions.shape[0]):
            pred_world.append(scene.make_world_coord_torch(predictions[i]))
        pred_world = torch.stack(pred_world)

        GT_world = scene.make_world_coord_torch(ground_truth)


        if metric_name == 'ADE':
            return ADE_best_of(predictions, ground_truth, metric_mask, obs_length)
        elif metric_name == 'FDE':
            return FDE_best_of(predictions, ground_truth, metric_mask, obs_length)
        if metric_name == 'ADE_world':
            return ADE_best_of(pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'FDE_world':
            return FDE_best_of(pred_world, GT_world, metric_mask, obs_length)
        else:
            raise ValueError("This metric has not been implemented yet!")

    def encode(self, inputs, if_test=False):
        x = inputs["x_augmented"].detach().clone() # T, B, 8
        num_agents = x.shape[1]

        # START SAMPLES LOOP
        all_context =[]
        all_aux_outputs = []
    
        ##################
        # PREDICT GOAL
        ##################
        tensor_image = inputs["tensor_image"].unsqueeze(0).repeat(num_agents, 1, 1, 1) 
        obs_traj_maps = inputs["input_traj_maps"][:, :self.args.obs_length]
        input_goal_module = torch.cat((tensor_image, obs_traj_maps), dim=1)
        goal_logit_map_start = self.goal_module(input_goal_module) # (num_agents, C+T, H, W) -> (num_agents, C_out, H, W), C_out = 12
        goal_prob_map = torch.sigmoid(goal_logit_map_start[:, -1:]) 
        if if_test:
            goal_point_start = TTST_test_time_sampling_trick(
                goal_prob_map, num_goals=self.args.num_samples, device=self.device)
            goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2) # final result: (num_agents, num_samples, 2)
        else:
            goal_point_start = x[-1,:,6:8].repeat(self.args.num_samples+1, 1, 1) 
            goal_point_start = goal_point_start.permute(1,0,2) # (num_agents, num_samples, 2)

        x_ori = x[self.args.obs_length-1,:,6:8]
        ################################
        # History Trajectory Encoding
        ################################
        if if_test:
            num_iters = self.args.num_samples+1
        else:
            num_iters = 1
        for sample_idx in range(num_iters):
            goal_point = goal_point_start[:, sample_idx].to(self.device) # (B,num_sample,2)->(B,2)
            goal_point = goal_point.detach()

            # agents trajectory up to now
            current_agents = torch.zeros([num_agents, self.args.obs_length, 8]).to(self.device)
            current_agents[:,:,2:] = x[:self.args.obs_length,:,:6].permute(1,0,2)
            current_agents[:,:,:2] = x[:self.args.obs_length,:,6:8].permute(1,0,2) - goal_point.unsqueeze(1)

            temporal_input_embedded = self.encoder.encode_hist(node_hist=current_agents, dropout_keep_prob=1) # [B, To, 2] -> [B, embedding_size], batch_first=True
            temporal_input_embedded = temporal_input_embedded.unsqueeze(1) # (B,1,embedding_size)


            aux_outputs = {
            "goal_logit_map": goal_logit_map_start,
            "goal_point": goal_point, # B, 2
            }

            all_context.append(temporal_input_embedded)
            all_aux_outputs.append(aux_outputs)

        all_context  = torch.stack(all_context)
        all_aux_outputs = {k: torch.stack([d[k] for d in all_aux_outputs])
                        for k in all_aux_outputs[0].keys()}
        return all_context, all_aux_outputs # (20,Tp+Tf,B,2) 

    def forward(self, inputs, if_test=False):
        x = inputs["x_augmented"].detach().clone()
        num_agents= x.shape[1]

        all_context, all_aux_outputs = self.encode(inputs, if_test=if_test) 

        vy = self.ts_sample(all_context=all_context) # [20, B, 1, 512] -> [20, B, Tf, 2]
        y = torch.zeros([self.args.num_samples, self.args.seq_length, num_agents, 2]).to(self.device)
        y[:, :self.args.obs_length] = x[:self.args.obs_length,:,6:8].repeat(self.args.num_samples, 1, 1, 1)
        # y[:, self.args.obs_length:] = vy.permute(0,2,1,3) # (20, Tp+Tf, B, 2) -> (20, B, Tp+Tf, 2)
        for i in range(self.args.obs_length, self.args.seq_length):
            y[:, i] = y[:, i-1] + vy[:, :, i-self.args.obs_length]
        
        return y, all_aux_outputs  # (20,Tp+Tf,B,2) 


    def get_loss(self, inputs, seq_list, t=None, if_test=False):
        all_context, all_aux_outputs = self.encode(inputs, if_test=if_test) 

        # compute goal BCE loss
        loss_mask = self.compute_loss_mask(seq_list, self.args.obs_length).to(self.device)
        out_maps_GT_goal = inputs["input_traj_maps"][:, self.args.obs_length:]
        goal_logit_map = all_aux_outputs["goal_logit_map"]
        goal_BCE_loss = Goal_BCE_loss(goal_logit_map, out_maps_GT_goal, loss_mask)

        # compute diffusion loss
        x = inputs["x_augmented"].detach().clone()
        vx_gt = x[self.args.obs_length:,:,2:4] # predict the velocity, (Tf, B, 2) 
        vx_gt = vx_gt.permute(1,0,2) # (B, Tf, 2)
        batch_size, _, point_dim = vx_gt.size() # (B, Tf, 2)
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        e_rand = torch.randn_like(vx_gt).cuda()  # (B, Tp, 2) 
        context = all_context[0]
        e_theta = self.diffnet(c0 * vx_gt + c1 * e_rand, beta=beta, context=context) 

        diffusion_loss = F.mse_loss(e_theta.reshape(-1, point_dim), e_rand.reshape(-1, point_dim), reduction='mean')
        
        losses = {
            "diffusion_loss": diffusion_loss,
            "goal_BCE_loss": goal_BCE_loss,
        }

        return losses

    def sample(self, all_context):
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        traj = {self.var_sched.num_steps: x_T} 
        all_outputs = []
        for t in range(self.var_sched.num_steps, self.determined_step, -1): # from x_T to 0
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            # print(t)
            # print(alpha_bar)
            sigma = self.var_sched.get_sigmas(t)

            c0 = 1.0 / torch.sqrt(alpha) # scalar
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar) # scalar
            x_t = traj[t]# result from previous denoising
            beta = self.var_sched.betas[[t]*batch_size]
            # print(beta)
            e_theta = self.diffnet(x_t, beta=beta, context=context) # predict noise, output: B * 12 * 2
            x_next = c0 * (x_t - c1 * e_theta) #+ sigma * z# compute the denoised x from x_t, whose dimension is [batch_size, num_points, point_dim]
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            del traj[t]
        middle_result = traj[self.determined_step]
        for sample_idx in range(self.args.num_samples):
            context = all_context[sample_idx]
            traj[self.determined_step] = middle_result
            for t in range(self.determined_step, 0, -1): # from x_T to 0
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                sigma = self.var_sched.get_sigmas(t)

                c0 = 1.0 / torch.sqrt(alpha) # scalar
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar) # scalar
                x_t = traj[t]# result from previous denoising
                beta = self.var_sched.betas[[t]*batch_size]
                e_theta = self.diffnet(x_t, beta=beta, context=context) # predict noise, output: B * 12 * 2
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z # compute the denoised x from x_t, whose dimension is [batch_size, num_points, point_dim]
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                del traj[t]
            all_outputs.append(traj[0])
            # del traj
        all_outputs = torch.stack(all_outputs)
        # all_intermediate_traj_outputs = traj # (diffusion_step, B, 12, 2), only the last sample
        return all_outputs # (B, 12, 2)

    def ddim_sample(self, all_context):
        ddim_step = int(20)
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) # generate guassian noise as initial, when t=T timestep
        ts = np.linspace(self.var_sched.num_steps, 0, (ddim_step + 1)) # (21,)
        all_outputs = []
        simple_var = False
        eta = 1
        if simple_var:
            eta = 1
        for mode_idx in range(self.args.num_samples):
            context = all_context[mode_idx]
            x_t = x_T
            for i in range(1,ddim_step + 1):
                cur_t = int(ts[i - 1]) #- 1 
                prev_t = int(ts[i]) #- 1 
                ab_cur = self.var_sched.alpha_bars[cur_t]
                ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
                beta = self.var_sched.betas[[cur_t]*batch_size]
                eps = self.diffnet(x_t, beta=beta, context=context)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x_t)

                first_term = (ab_prev / ab_cur)**0.5 * x_t
                second_term = ((1 - ab_prev - var)**0.5 -
                                (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x_t = first_term + second_term + third_term
            all_outputs.append(x_t)
            # del traj
        all_outputs = torch.stack(all_outputs)
        return all_outputs # (batch_size, num_points, point_dim) = (B, 12, 2)

    def ddpm_sample(self, all_context):
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        simple_var = False
        all_outputs = []
        for mode_idx in range(self.args.num_samples):
            context = all_context[mode_idx]
            traj = {self.var_sched.num_steps: x_T} 
            x_t = x_T
            for t in range(self.var_sched.num_steps , 0, -1):
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                # print([t]*batch_size)
                beta = self.var_sched.betas[[t]*batch_size]
                # beta = self.var_sched.betas[t]
                # print(beta)
                # beta = torch.full((batch_size,), beta, device=context.device, dtype=torch.long)
                if t == 0:
                    noise = 0
                else:
                    if simple_var:
                        var = self.var_sched.betas[t]
                    else:
                        var = (1 - self.var_sched.alpha_bars[t - 1]) / (
                            1 - self.var_sched.alpha_bars[t]) * self.var_sched.betas[t]
                    noise = torch.randn_like(x_t)
                    noise *= torch.sqrt(var)
                x_t = traj[t]
                eps = self.diffnet(x_t, beta=beta, context=context)

                x_next = (x_t -
                        (1 - alpha) / torch.sqrt(1 - alpha_bar) *
                        eps) / torch.sqrt(alpha) + noise
                traj[t-1] = x_next.detach()
                traj[t] = traj[t].cpu()
                del traj[t]
            all_outputs.append(traj[0])
        all_outputs = torch.stack(all_outputs)
        return all_outputs

    def ddim_ts_sample(self, all_context):
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        # print(x_T.shape)
        all_outputs = []
        ddim_step = int(20)
        determined_step = int(8)
        ts = np.linspace(self.var_sched.num_steps, 0, (ddim_step + 1)) # (21,)
        simple_var = False
        eta = 0
        if simple_var:
            eta = 1
        x_t = x_T
        for i in range(1,determined_step+1):
            cur_t = int(ts[i - 1]) #- 1 
            prev_t = int(ts[i]) #- 1 
            ab_cur = self.var_sched.alpha_bars[cur_t]
            ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
            beta = self.var_sched.betas[[cur_t]*batch_size]
            eps = self.diffnet(x_t, beta=beta, context=context)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x_t)

            first_term = (ab_prev / ab_cur)**0.5 * x_t
            second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x_t = first_term + second_term + third_term
        middle_result = x_t
        for sample_idx in range(self.args.num_samples):
            context = all_context[sample_idx]
            x_t = middle_result
            for i in range(determined_step+1,ddim_step+1):
                cur_t = int(ts[i - 1]) #- 1 
                prev_t = int(ts[i]) #- 1 
                ab_cur = self.var_sched.alpha_bars[cur_t]
                ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
                beta = self.var_sched.betas[[cur_t]*batch_size]
                eps = self.diffnet(x_t, beta=beta, context=context)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x_t)

                first_term = (ab_prev / ab_cur)**0.5 * x_t
                second_term = ((1 - ab_prev - var)**0.5 -
                                (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x_t = first_term + second_term + third_term
            all_outputs.append(x_t)
            # del traj
        all_outputs = torch.stack(all_outputs)
        return all_outputs # (B, 12, 2)
    
    def ts_sample(self, all_context):
        all_outputs = []

        # trunk stage 
        context = all_context[-1]
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.args.pred_length, 2]).to(context.device) 
        x_t = x_T
        for t in range(self.var_sched.num_steps, self.args.trunk_stage_step, -1):
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]

            c0 = 1.0 / torch.sqrt(alpha) # scalar
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar) # scalar
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.diffnet(x_t, beta=beta, context=context) 
            x_next = c0 * (x_t - c1 * e_theta) 
            x_t = x_next     
        
        # branch stage
        middle_result = x_t
        simple_var = True if self.args.dataset in ['eth5', 'ind'] else False
        eta = 0
        if simple_var:
            eta = 1
        ts = np.linspace(self.var_sched.num_steps, 0, (self.args.ddim_step + 1)) # (21,)
        for sample_idx in range(self.args.num_samples):
            context = all_context[sample_idx]
            x_t = middle_result
            
            for i in range(int(self.args.branch_stage_step + 1), int(self.args.ddim_step + 1)):
                cur_t = int(ts[i - 1]) #- 1 
                prev_t = int(ts[i]) #- 1 
                ab_cur = self.var_sched.alpha_bars[cur_t]
                ab_prev = self.var_sched.alpha_bars[prev_t] if prev_t >= 0 else 1
                beta = self.var_sched.betas[[cur_t] * batch_size]
                eps = self.diffnet(x_t, beta=beta, context=context)
                var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x_t)

                first_term = (ab_prev / ab_cur)**0.5 * x_t
                second_term = ((1 - ab_prev - var)**0.5 -
                                (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
                x_t = first_term + second_term + third_term

            all_outputs.append(x_t) 
        all_outputs = torch.stack(all_outputs)
        # all_intermediate_traj_outputs = traj # (diffusion_step, B, 12, 2), only the last sample
        return all_outputs # (B, 12, 2)