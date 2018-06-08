# CartPole
A Deep Q Network used for the CartPole-v0 task from the [OpenAI Gym](https://gym.openai.com/).
It is mostly similar to the one described in this [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
with a change to the reward that passes it through a tanh() function to limit its range and make it smoother and a change to the optimization part
that uses the equations from [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).

After 2000 episodes of training, the model can manage to balance the pole for a bit:

![CartPole Animation](static/animation.gif)

