import torch
from util import *

def simple_model():
    model = Model(vocab_size=3, embedding_dim=10, prompt_length=3, response_length=3)
    prompts = torch.tensor([[1, 0, 2]])  # [batch pos]
    torch.manual_seed(10)

    responses = generate_responses(prompts=prompts, model=model, num_responses=2)  # [batch trial pos]  @inspect responses
    print("responses:", responses)
    
    rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=sort_inclusion_ordering_reward)  # [batch trial]  @inspect rewards
    print("rewards（奖励）:", rewards)

    deltas = compute_deltas(rewards=rewards, mode="rewards")  # [batch trial]  @inspect deltas
    print("deltas（rewards模式）:", deltas)

    deltas_centered = compute_deltas(rewards=rewards, mode="centered_rewards")  # [batch trial]  @inspect deltas
    print("deltas（centered_rewards模式）:", deltas_centered)

    deltas_normalized = compute_deltas(rewards=rewards, mode="normalized_rewards")  # [batch trial]
    print("deltas（normalized_rewards模式）:", deltas_normalized)

    deltas_max = compute_deltas(rewards=rewards, mode="max_rewards")  # [batch trial]  @inspect deltas
    print("deltas（max_rewards模式）:", deltas_max)

    log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]  @inspect log_probs
    print("log_probs（对数概率）:", log_probs)

    loss_naive = compute_loss(log_probs=log_probs, deltas=deltas, mode="naive")  # @inspect loss
    print("loss（naive模式）:", loss_naive)

    old_model = Model(vocab_size=3, embedding_dim=10, prompt_length=3, response_length=3)  # Pretend this is an old checkpoint @stepover
    old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=old_model)  # @stepover

    loss_unclipped = compute_loss(log_probs=log_probs, deltas=deltas, mode="unclipped", old_log_probs=old_log_probs)  # @inspect loss
    print("loss（unclipped模式）:", loss_unclipped)

    loss_clipped = compute_loss(log_probs=log_probs, deltas=deltas, mode="clipped", old_log_probs=old_log_probs)  # @inspect loss
    print("loss（clipped模式）:", loss_clipped)


if __name__ == "__main__":
    simple_model()