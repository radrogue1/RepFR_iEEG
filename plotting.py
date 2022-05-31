import matplotlib.pyplot as plt
def set_axes(figsize = (7,7), xlim = None, ylim = None, xticks = None, yticks = None, fontsize = 14):
    fig, ax = plt.subplots(figsize=figsize)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    _ = plt.xticks(xticks, fontsize = fontsize)
    _ = plt.yticks(yticks, fontsize = fontsize)
    if xlim:
        _ = plt.xlim(xlim)
    if ylim:
        _ = plt.ylim(ylim)
    _ = plt.xlabel(None, fontsize = fontsize)
    _ = plt.ylabel(None, fontsize = fontsize)
    return (fig,ax)