# Space Invaders
A Deep Q Network trained to play Space Invaders using [OpenAI Gym](https://gym.openai.com/).
It is mostly similar to the one described in this [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
with a change to the reward that passes it through a tanh() function to limit its range and make it smoother and a change
to the optimization part that uses the equations from [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
and a final change to the action policy, where we use a Bolzmann exploration approach instead of the more popular ε-greedy approach.
