import torch
import torch.nn as nn


def MSE_loss(outputs, ground_truth, loss_mask):
    """
    Compute an averaged Mean-Squared-Error, only on the positions
    in which loss_mask is True.
    outputs.shape = num_samples, seq_length, num_agent, n_coords(2)
    """
    squared_error = (outputs - ground_truth)**2
    # squared_error = (outputs[:-1] - ground_truth[:-1])**2
    # squared_error = squared_error + 5 * (outputs[-1] - ground_truth[-1])**2
    # sum over coordinates --> Shape becomes: num_samples * seq_len * n_agents
    squared_error = torch.sum(squared_error, dim=-1)

    # compute error only where mask is True
    loss = squared_error * loss_mask

    # Take a weighted loss, but only on places where loss_mask=True.
    # Divide by loss_mask.sum() instead of seq_len*N_pedestrians (or mean).
    # This is an average loss per time-step per pedestrian.
    loss = loss.sum(dim=-1).sum(dim=-1) / loss_mask.sum()

    # minimum loss over samples (only 1 sample during training)
    loss, _ = loss.min(dim=0)

    return loss


def Goal_BCE_loss(logit_map, goal_map_GT, loss_mask):
    """
    Compute the Binary Cross-Entropy loss for the probability distribution
    of the goal. Prediction and GT are two pixel maps.
    """
    losses_samples = []
    for logit_map_sample_i in logit_map:
        loss = BCE_loss_sample(logit_map_sample_i, goal_map_GT, loss_mask)
        losses_samples.append(loss)
    losses_samples = torch.stack(losses_samples)

    # minimum loss over samples (only 1 sample during training)
    loss, _ = losses_samples.min(dim=0)

    return loss


def BCE_loss_sample(logit_map, goal_map_GT, loss_mask):
    """
    Compute the Binary Cross-Entropy loss for the probability distribution
    of the goal maps. logit_map is a logit map, goal_map_GT a probability map.
    """
    batch_size, T, H, W = logit_map.shape
    # reshape across space and time
    output_reshaped = logit_map.view(batch_size, -1)
    target_reshaped = goal_map_GT.view(batch_size, -1)

    # takes as input computed logit and GT probabilities
    BCE_criterion = nn.BCEWithLogitsLoss(reduction='none')

    # compute the Goal CE loss for each agent and sample
    loss = BCE_criterion(output_reshaped, target_reshaped)

    # Mean over maps (T, W, H)
    loss = loss.mean(dim=-1)

    # Take a weighted loss, but only on places where loss_mask=True
    # Divide by full_agents.sum() instead of seq_len*N_pedestrians (or mean)
    full_agents = loss_mask[-1]
    loss = (loss * full_agents).sum(dim=0) / full_agents.sum()

    return loss

def BCE_loss_sample_goal(logit_map, goal_map_GT, loss_mask):
    """
    Compute the Binary Cross-Entropy loss for the probability distribution
    of the goal maps. logit_map is a logit map, goal_map_GT a probability map.
    """
    batch_size, T, H, W = logit_map.shape
    # reshape across space and time
    output_reshaped = logit_map.view(batch_size, -1)
    target_reshaped = goal_map_GT.view(batch_size, -1)
    output_goal = output_reshaped[:, -1:]
    target_goal = target_reshaped[:, -1:]
    output_reshaped = output_reshaped[:, :-1]
    target_reshaped = target_reshaped[:, :-1]
    # takes as input computed logit and GT probabilities
    BCE_criterion = nn.BCEWithLogitsLoss(reduction='none')

    # compute the Goal CE loss for each agent and sample
    loss = BCE_criterion(output_reshaped, target_reshaped)
    loss_goal = BCE_criterion(output_goal, target_goal)
    loss = loss + 5 * loss_goal

    # Mean over maps (T, W, H)
    loss = loss.mean(dim=-1)

    # Take a weighted loss, but only on places where loss_mask=True
    # Divide by full_agents.sum() instead of seq_len*N_pedestrians (or mean)
    full_agents = loss_mask[-1]
    loss = (loss * full_agents).sum(dim=0) / full_agents.sum()

    return loss

def wta_loss(prediction, gt, loss_mask):
    '''
    prediction: predicted forecasts, of shape [batch, hypotheses, 2]
    gt: ground-truth forecasting trajectory, of shape [batch, 2]
    gt_valid_mask: ground-truth forecasting mask indicating the valid future steps, of shape [batch]
    
    '''
    # compute the L2 distance between each hypothesis and the ground truth
    distance = torch.norm(prediction - gt.unsqueeze(1), p=2, dim=2) / loss_mask.unsqueeze(1) # [batch, hypotheses]
    # distance = distance.replace(0, float('inf')) # set the distance to infinity if the ground truth is not valid
    # select the prediction with the minimum distance to the ground truth
    
    # distance = torch.norm(prediction - gt.unsqueeze(0), p=2, dim=2) # FDE (20,B)
    # full_agents = loss_mask[:,-1] # (20,B)
    # losses_samples = (error * full_agents).sum(dim=1) / full_agents.sum(dim=1) # 20
    # goal_FDE_loss, idx_samples = losses_samples.min(dim=0) # 
    
    # select the prediction with the minimum distance to the ground truth
    nearest_hypothesis_idxs = distance.argmin(dim=-1) # [batch]
    nearest_hypothesis_bs_idxs = torch.arange(nearest_hypothesis_idxs.shape[0]).type_as(nearest_hypothesis_idxs) # [batch]
    
    # extract the L2 distance between the selected hypothesis and gt
    loss_reg = distance[nearest_hypothesis_bs_idxs, nearest_hypothesis_idxs] # [batch]
    return loss_reg.mean(), nearest_hypothesis_idxs, nearest_hypothesis_bs_idxs # mean over the batch