import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(rewards)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def plotWithRewards(scores, mean_scores, rewards):
    figure, (ax1, ax2) = plt.subplots(2, sharex=True) 
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    # plt.title('Training...')
    # ax1.xlabel('Number of Games')
    # ax1.ylabel('Score')

    figure.suptitle('Scores')
    ax1.plot([1, 2], scores)
    ax1.plot([1, 2], mean_scores)
    # ax1.ylim(ymin=0)
    # ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    # ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    ax2.plot([1, 2], rewards)
    # ax2.xlabel('Frames')
    # ax2.ylabel('Reward')

    # plt.show(block=False)
    plt.show(block=True)
    plt.pause(.1)

# plot([102, 106], [102, 104], [6, 7])