import torch
import os
import sys
from tqdm import tqdm
from typing import Callable
from matplotlib import pyplot as plt
from util import *
import copy



def run_policy_gradient(num_epochs: int = 100,
                        num_steps_per_epoch: int = 10,
                        compute_ref_model_period: int = 10,
                        num_responses: int = 10,
                        deltas_mode: str = "rewards",
                        loss_mode: str = "naive",
                        kl_penalty: float = 0.0,
                        reward_fn: Callable[[torch.Tensor, torch.Tensor], float] = sort_inclusion_ordering_reward,
                        use_cache: bool = False):
    """Train a model using policy gradient.
    Return:
    - Path to the image of the learning curve.
    - Path to the log file
    """
    torch.manual_seed(42)

    image_path = f"policy_gradient_{deltas_mode}_{loss_mode}.png"
    
    
    # Define the data
    prompts = torch.tensor([[1, 0, 2], [3, 2, 4], [1, 2, 3]])
    vocab_size = int(prompts.max().item()) + 1
    prompt_length = response_length = prompts.shape[1]
    
    model = Model(vocab_size=vocab_size, embedding_dim=10, prompt_length=prompt_length, response_length=response_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    records = []
    ref_log_probs = None
    ref_model = None
    old_log_probs = None
    
    
    for epoch in tqdm(range(num_epochs), desc="epoch"):
        # If using KL penalty, need to get the reference model (freeze it every few epochs)
        if kl_penalty != 0:
            if epoch % compute_ref_model_period == 0:
                ref_model = copy.deepcopy(model)
        
        # Sample responses and evaluate their rewards
        responses = generate_responses(prompts=prompts, model=model, num_responses=num_responses)  # [batch trial pos]
        rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=reward_fn)  # [batch trial]
        deltas = compute_deltas(rewards=rewards, mode=deltas_mode)  # [batch trial]
        
        if kl_penalty != 0:  # Compute under the reference model
            with torch.no_grad():
                ref_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=ref_model)  # [batch trial]
        
        if loss_mode != "naive":  # Compute under the current model (but freeze while we do the inner steps)
            with torch.no_grad():
                old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]
        
        # Take a number of steps given the responses
        for step in range(num_steps_per_epoch):
            log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]
            loss = compute_loss(log_probs=log_probs, deltas=deltas, mode=loss_mode, old_log_probs=old_log_probs)  # @inspect loss
            if kl_penalty != 0:
                loss += kl_penalty * compute_kl_penalty(log_probs=log_probs, ref_log_probs=ref_log_probs)
            
            # Print information
            # print_information(epoch=epoch, step=step, loss=loss, prompts=prompts, rewards=rewards, responses=responses, 
            #     log_probs=log_probs, deltas=deltas)
            global_step = epoch * num_steps_per_epoch + step
            records.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "mean_reward": rewards.mean().item()})
            
            # Backprop and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Plot step versus loss and reward in two subplots
    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    rewards = [r["mean_reward"] for r in records]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss subplot
    ax1.plot(steps, losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Train Loss")
    
    # Reward subplot
    ax2.plot(steps, rewards)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Mean Reward")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return image_path




if __name__ == "__main__":
    
    image_path = run_policy_gradient(num_epochs=100, num_steps_per_epoch=10, num_responses=10, deltas_mode="centered_rewards", loss_mode="naive", 
        kl_penalty=0.1, reward_fn=sort_inclusion_ordering_reward)
    print("image_path:", image_path)