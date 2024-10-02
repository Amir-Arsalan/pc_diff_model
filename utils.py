import os, pickle, random, time
import numpy as np
from pathlib import Path
from plotClass import plotClass

def fileExist(path):
    if path != '/':
        if os.path.isdir(path):
            return True
        else:
            temp = Path(path)
            return temp.is_file()
    else:
        return False

def mkdirs(paths):
        try:
            if isinstance(paths, list) and not isinstance(paths, str):
                for path in paths:
                    mkdir(path)
            else:
                mkdir(paths)
        except:
            time.sleep(random.random()/5)
            if isinstance(paths, list) and not isinstance(paths, str):
                for path in paths:
                    mkdir(path)
            else:
                mkdir(paths)

def mkdir(path):
    if not fileExist(path):
        os.makedirs(path)


def savePickle(filePath, data, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filePath, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)

def loadPickle(filePath):
    data = None
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    return data
    
def safePickleLoad(filePath):
    try:
        if fileExist(filePath):
            loadedPkl = loadPickle(filePath)
        else:
            loadedPkl = False
    except:
        loadedPkl = False
    return loadedPkl

def plot(x_label, y_label, min_x, max_x, min_y, max_y, X, Y, file_name, X_num_ticks=11, Y_num_ticks=11, hide_X_ticks=False, hide_Y_ticks=False, legend_labels=[], include_in_legend=[], line_styles=[], markers=[], colors=[], y_grid=True):
    # X & Y must be a list of lists
    plot = plotClass()

    # Values
    figIndex = plot.createSubplot(figSize=8.4)
    axArgs = plot.figures['ax_kwargs'][figIndex]
    plotParams = plot.figures['plotParams'][figIndex]

    margin=0.01
    axArgs['xLimLow'] = min_x-margin
    axArgs['xLimHigh'] = max_x+margin
    axArgs['yLimLow'] = min_y-margin
    axArgs['yLimHigh'] = max_y+margin
    # axArgs['xTicksMajor'] = np.array([math.log(alpha) for alpha in alpha_values][::40] + [math.log(alpha_values[-1])])
    axArgs['xTicksMajor'] = np.linspace(min_x, max_x, X_num_ticks)
    axArgs['yTicksMajor'] = np.linspace(min_y, max_y, Y_num_ticks)
    if not hide_X_ticks:
        axArgs['xTickLabelsMajor'] = [('%d' if isinstance(max_x, int) else '%.2f') % xTick for xTick in axArgs['xTicksMajor']]
    if not hide_Y_ticks:
        axArgs['yTickLabelsMajor'] = [('%d' if isinstance(max_y, int) else '%.2f') % yTick for yTick in axArgs['yTicksMajor']]
    axArgs['xTickFontSize'] = 12
    axArgs['yTickFontSize'] = 12
    axArgs['xTickRot'] = 0
    axArgs['xLabel'] = x_label
    axArgs['yLabel'] = y_label
    axArgs['xLabelFontSize'] = 14
    axArgs['yLabelFontSize'] = 14
    axArgs['xLabelWeight'] = 'bold'
    axArgs['yLabelWeight'] = 'bold'
    axArgs['yGrid'] = y_grid
    axArgs['plotSavePath'] = '{0}.png'.format(file_name)
    plotParams['x'] = X
    plotParams['y'] = Y

    legend_kwargs = {'markerscale': 0.6, 'loc': 'upper left', 'prop': {'size': 11}}

    lineProps = []
    for i in range(len(X)):
        # lineProps.append({'color': colors[i], 'linestyle': line_styles[i], 'marker': markers[i], 'markersize': 20})
        lineProps.append({'color': colors[i], 'linestyle': line_styles[i], 'marker': markers[i]})
    
    plotParams['lineProps'] = lineProps

    if not legend_labels:
        plot.plotPoints(figIdx=figIndex, axType='plot')
    else:
        plot.plotPoints(figIdx=figIndex, axType='plot', axLegend=True, legend_labels=legend_labels, legend_kwargs=legend_kwargs)
    plot.savePlot(figIdx=figIndex)