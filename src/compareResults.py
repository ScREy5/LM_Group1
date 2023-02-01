import matplotlib.pyplot as plt
import csv

def get_x_y(data):
    data_list = data[0].split(',')
    X = []
    Y = []
    for i in range(len(data_list)):
        X.append(i)
        if len(Y) == 0:
            val = 0 if data_list[i] != 1 else 1
            Y.append(val)
        elif data_list[i] == '1':
            Y.append(Y[-1] + 1)
        else:
            Y.append(Y[-1])
    return X, Y

def plot_rewards(data_path, save_path):
    arenas = ['Train', 'Val1', 'Val2', 'Val3', 'Val4']
    for arena in arenas:
        X = [0, 0, 0, 0, 0, 0, 0, 0]
        Y = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(29):
            food = f"run_{i+1}_val_food.txt"
            if i < 19:
                with open(data_path + "Josip/" + arena + f"/{food}") as file:
                    f = file.readlines()
                    for line in f:
                        X[int(eval(line))] += 1
            else:
                with open(data_path + "Gary/" + arena + f"/run_{i-18}_val_food.txt") as file:
                    f = file.readlines()
                    for line in f:
                        X[int(eval(line))] += 1

        for i in range(1, len(X)):
            X[i] += X[i-1]
            X[i-1] = X[i-1]/30
        X[-1] = X[-1]/30
        plt.clf()
        plt.title('DQN Algorithm')
        plt.xlabel('steps taken')
        plt.ylabel('Food eaten')
        ax = plt.gca()
        ax.set_xlim([0, 60])
        plt.plot(X, Y)
        plt.savefig(save_path + f"{arena}.png")

if __name__ == '__main__':
    # For model v2 testing results
    # data = "testing_results/action_list_full_v2_scaleChange_episode_350"
    # save = "graphs/steps_graph.png"


    # For model v2 validation run:
    data = "C:/Users/Josip/PycharmProjects/learningmachines/LM_Group1/src/logs/task2/validation_runs/"
    save = "steps_food_graph"
    plot_rewards(data, save)