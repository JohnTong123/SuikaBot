import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def plotWithRewards(scores, mean_scores, rewards, mean_rewards):
    if len(rewards) != 1:
        plt.close()  # Close the previous plot, if any
    figure, (ax1, ax2) = plt.subplots(2, sharex=True) 
    display.clear_output(wait=True)
    # plt.clf()
    # display.display(plt.gcf())
    # plt.clf()

    # Set titles for each subplot and the figure
    ax1.set_title('Scores over Time')
    ax2.set_title('Rewards over Time')
    figure.suptitle('Training Progress')

    # Plotting the scores and mean scores on ax1
    ax1.plot(scores, label='Scores')
    ax1.plot(mean_scores, label='Mean Scores')
    ax1.legend()

    # Optional: Annotate the last point for better clarity
    ax1.annotate(f'{scores[-1]}', (len(scores)-1, scores[-1]), textcoords="offset points", xytext=(0,10), ha='center')
    ax1.annotate(f'{mean_scores[-1]}', (len(mean_scores)-1, mean_scores[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    ax2.annotate(f'{rewards[-1]}', (len(rewards)-1, rewards[-1]), textcoords="offset points", xytext=(0,10), ha='center')
    ax2.annotate(f'{mean_rewards[-1]}', (len(mean_rewards)-1, mean_rewards[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    # Plotting the rewards on ax2
    ax2.plot(rewards, label='Rewards', color='green')
    ax2.plot(mean_rewards, label='Mean Rewards')
    ax2.legend()

    # Setting labels for axes
    ax2.set_xlabel('Number of Games')
    ax1.set_ylabel('Score')
    ax2.set_ylabel('Reward')

    # Display the plot
    
    plt.show(block=False)
    plt.pause(0.1)


# plot([102, 106], [102, 104], [6, 7])
