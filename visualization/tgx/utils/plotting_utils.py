import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from typing import List, Union
import matplotlib

def create_ts_list(start, end, metric=None, interval=None):
    if metric == "Unix" or metric == "unix" or metric == "UNIX":
        start = datetime.datetime.fromtimestamp(start).date()
        end = datetime.datetime.fromtimestamp(end).date()
        if interval == 'daily':
            date_list = pd.date_range(start = start, end = end, freq="D") 
        elif interval == "month":
            date_list = pd.date_range(start = start, end = end, freq="M")
        elif interval == "year":
            date_list = pd.date_range(start = start, end = end, freq="Y") 
        timelist = []
        for dates in date_list:
            timelist.append(dates.strftime("%Y/%m/%d"))
    else:
        timelist = list(range(start, end, interval))
    # print(timelist)
    return timelist
    

def plot_nodes_edges_per_ts(edges: list,
                            nodes: list,
                            date_ls: Union[list, None] = None,
                            ts: list = [],
                            title: str = "",
                            filename: str = None,
                            ylabel_1: str = 'Edges per Timestamp',
                            ylabel_2: str = 'Nodes per Timestamp',
                            fig_size: tuple = (7,5),
                            font_size: int = 20,
                            ticks_font_size: int = 15):
    """
    Plot nodes and edges per timestamp in one figure
    Parameters:
        edges: A list containing number of edges per timestamp
        nodes: A list containing number of nodes per timestamp
        ts: list of timestamps
        filename: Name of the output file name, containing the path
        ylabel_1: Label for the edges per timestamp line
        ylabel_2: Label for the nodes per timestamp line
    """
    if date_ls is None:
        fig = plt.figure(facecolor='w', figsize=fig_size)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        c1, = ax1.plot(ts, edges, color='royalblue', lw=2, label=ylabel_1)
        c2, = ax2.plot(ts, nodes, color='goldenrod', linestyle='dashed', lw=2, label=ylabel_2)
        curves = [c1, c2]
        ax1.legend(curves, [curve.get_label() for curve in curves], fontsize = ticks_font_size)
        ax1.set_xlabel('Time', fontsize=font_size)
        ax1.set_ylabel(ylabel_1, fontsize=font_size)
        ax2.set_ylabel(ylabel_2, fontsize=font_size)
        ax1.tick_params(labelsize=ticks_font_size)
        ax2.tick_params(labelsize=ticks_font_size)
        ax1.set_ylim(0)
        ax2.set_ylim(0)
        ax1.set_xlim(0, len(ts)-1)
        ax1.set_title(title, fontsize=font_size)
        if filename is not None:
            plt.savefig(f'{filename}')
        else:
            plt.show()
    
    else:
        ts = date_ls
        fig = plt.figure(facecolor='w', figsize=fig_size)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        c1, = ax1.plot(ts, edges, color='royalblue', lw=2, label=ylabel_1)
        c2, = ax2.plot(ts, nodes, color='goldenrod', lw=2, label=ylabel_2)
        curves = [c1, c2]
        ax1.legend(curves, [curve.get_label() for curve in curves], title= "Legend")
        ax1.set_xlabel('Time', fontsize=font_size)
        ax1.set_ylabel(ylabel_1, fontsize=font_size)
        ax2.set_ylabel(ylabel_2, fontsize=font_size)
        ax1.tick_params(labelsize=ticks_font_size)
        ax2.tick_params(labelsize=ticks_font_size)
        ax1.set_ylim(0)
        ax2.set_ylim(0)
        ax1.tick_params(axis='x', rotation=70)
        locator = matplotlib.ticker.MaxNLocator(nbins=10)
        ax1.xaxis.set_major_locator(locator)
        ax1.set_title(title, fontsize=font_size)

        if filename is not None:
            plt.savefig(f'{filename}')
        else:
            plt.show()

def plot_for_snapshots(data: list = [],
                       date_ls: Union[list, None] = None,
                       title: str = "",
                       y_title: str = "step",
                       filename: str = None, 
                       show_ave: bool=True, 
                       fig_size: tuple = (7,5),
                       font_size: int = 20,
                       ticks_font_size: int = 15):
    '''
    Plot a variable for different timestamps
    Parameters:
        data: A list of desired variable to be plotted
        y_title: Title of the y axis
        filename: Name of the output file name, containing the path
        show_ave: Whether to plot a line showing the average of the variable over all timestamps
    '''
    plt.style.use('fivethirtyeight')
    
    if date_ls is None:
        ts = list(range(0, len(data)))
        # plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(facecolor='w', figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.plot(ts, data, color='royalblue', lw=2)

        ax.set_xlabel('Time', fontsize=font_size)
        ax.set_ylabel(y_title, fontsize=font_size)
        ax.tick_params(labelsize=ticks_font_size)
        ax.set_xlim(0, len(ts)-1)
        ax.set_title(title)
        if show_ave:
            ave_deg = [np.average(data) for i in range(len(ts))]
            ax.plot(ts, ave_deg, color='#ca0020', linestyle='dashed', lw=3)
        if filename is not None:
            plt.savefig(f'{filename}')
        else:
            plt.show()
    
    else:
        ts = date_ls
        # plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(facecolor='w', figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.plot(ts, data, color='royalblue', lw=2)

        ax.set_xlabel('Time', fontsize=font_size)
        ax.set_ylabel(y_title, fontsize=font_size)
        ax.tick_params(labelsize=ticks_font_size)
        ax.tick_params(axis='x', rotation=70)
        locator = matplotlib.ticker.MaxNLocator(nbins=10)
        ax.xaxis.set_major_locator(locator)
        ax.set_title(title, fontsize=font_size)
        if show_ave:
            ave_deg = [np.average(data) for i in range(len(ts))]
            ax.plot(ts, ave_deg, color='#ca0020', linestyle='dashed', lw=2)
        if filename is not None:
            plt.savefig(f'{filename}')
        else:
            plt.show()


def plot_density_map(data: list,
                     date_ls: Union[list, None] = None,
                     y_title: str = None,
                     filename: str = None,
                     fig_size: tuple = (7,5),
                     font_size: int = 20,
                     ticks_font_size: int = 15):
    '''
    Plot a density map using fig and ax
    Parameters:
        data: A list of desired variable to be plotted
        y_title: Title of the y axis
        filename: Name of the output file name, containing the path
    '''
    # plt.style.use('_classic_test_patch')
    
    if date_ls is None:
        max_value = max(max(inner) for inner in data if inner)
        c = np.zeros((max_value, len(data)))

        for i, row in enumerate(data):
            for value in row:
                c[value - 1][i] += 1

        # Plot
        fig = plt.figure(facecolor='w', figsize=fig_size)
        ax = fig.add_subplot(111)

        norm = mcolors.Normalize(vmin=0, vmax=1)
        cax = ax.imshow(c, cmap='viridis', interpolation='nearest', norm=norm)
        cbar = fig.colorbar(cax)
        cbar.set_label('Frequency')

        ax.set_title("Heatmap of Node Degrees Over Time")
        ax.set_xlabel('Time', fontsize=font_size)
        ax.set_ylabel(y_title, fontsize=font_size)
        ax.tick_params(labelsize=ticks_font_size)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Adjust the aspect ratio of the plot
        ax.set_aspect('auto')

        if filename is not None:
            plt.savefig(f'{filename}')
        else:
            plt.show()
    
    else:
        max_value = max(max(inner) for inner in data if inner)
        c = np.zeros((max_value, len(data)))

        for i, row in enumerate(data):
            for value in row:
                c[value - 1][i] += 1

        # Plot
        fig = plt.figure(facecolor='w', figsize=fig_size)
        ax = fig.add_subplot(111)

        norm = mcolors.Normalize(vmin=0, vmax=1)
        cax = ax.imshow(c, cmap='viridis', interpolation='nearest', norm=norm)
        cbar = fig.colorbar(cax)
        cbar.set_label('Frequency')

        ax.set_title("Heatmap of Node Degrees Over Time")
        ax.set_xlabel('Time', fontsize=font_size)
        ax.set_ylabel(y_title, fontsize=font_size)
        
        ax.tick_params(axis='x', rotation=70, )
        ax.set_xticks(range(0, len(date_ls)), date_ls)
        locator = matplotlib.ticker.MaxNLocator(nbins=10)
        ax.xaxis.set_major_locator(locator)

        # Adjust the aspect ratio of the plot
        ax.set_aspect('auto')

        if filename is not None:
            plt.savefig(f'{filename}')
        else:
            plt.show()


if __name__ == "__main__":
    create_ts_list(86400, 86400*365, "unix", "month")
    create_ts_list(2015, 2022, interval=2)