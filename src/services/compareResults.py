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
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        lines = []
        line_count = -1
        for line in csv_reader:
            line_count += 1
            if line_count % 2:
                continue
            lines.append(line)



    plt.figure(1)
    rewards_sum = lines[-1]
    X, Y = get_x_y(rewards_sum)
    plt.clf()
    plt.title('DQN Algorithm 1'
              '')
    plt.xlabel('steps taken')
    plt.ylabel('straight movement')
    ax = plt.gca()
    ax.set_xlim([0, X[-1]])
    ax.set_ylim([0, X[-1]])
    plt.plot(X, Y)
    plt.savefig(save_path)

if __name__ == '__main__':
    # For model v2 testing results
    # data = "testing_results/action_list_full_v2_scaleChange_episode_350"
    # save = "graphs/steps_graph.png"


    # For model v2 validation run:
    data = "testing_results/action_list_full_v1_validation"
    save = "graphs/steps_graph_validation_run_v1_model.png"
    plot_rewards(data, save)