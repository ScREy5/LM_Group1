import matplotlib.pyplot as plt
import numpy as np

run = 3
dqn1 = np.loadtxt(f"logs/dqn1/val_dq_{run}.txt").astype(int)
dqn1_actions = np.equal(dqn1, 1).astype(int)
plt.plot(np.cumsum(dqn1_actions),label = "DQN algorithm 1")
plt.scatter(len(dqn1_actions),np.cumsum(dqn1_actions)[-1],c="red", s=10)
dqn2 = np.genfromtxt(f"logs/dqn2/val_dq_{run}.csv", delimiter=",").astype(int)
dqn2_actions = np.equal(dqn2, 1).astype(int)
plt.plot(np.cumsum(dqn2_actions),label = "DQN algorithm 2")
plt.scatter(len(dqn2_actions),np.cumsum(dqn2_actions)[-1],c="red", s=10)
tqn = np.genfromtxt(f"logs/tabq/val_q_{run}.csv", delimiter=",").astype(int)
tqn_actions = np.equal(tqn, 1).astype(int)
plt.plot(np.cumsum(tqn_actions),label = "Tabular Q")
plt.scatter(len(tqn_actions),np.cumsum(tqn_actions)[-1],c="red", s=10)


plt.xlim([0,250])
plt.ylim([0,250])
plt.legend()
plt.suptitle(f"Performance on arena {run}")
plt.ylabel("Straight movements")
plt.xlabel("Steps taken")
plt.show()
print(np.cumsum(dqn1_actions)[-1]/len(dqn1_actions))
print(np.cumsum(dqn2_actions)[-1]/len(dqn2_actions))
print(np.cumsum(tqn_actions)[-1]/len(tqn_actions))
