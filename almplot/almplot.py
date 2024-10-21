from sys import argv
import matplotlib
from matplotlib.scale import SymmetricalLogTransform

# matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.ticker import ScalarFormatter

'''
Global constants
'''
ZERO = 0.0
INFTY = 1e50

ALAMO_TIMEOUT = 68
ALAMO_SUCCESS = 0

'''
Global options
'''
show_invalid_value = False

'''
Utility functions
'''
def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def stylized_boxplot(data, ax=None):
    if ax is None:
        ax = plt.gca()
    plot_data = data
    if len(data) > 1:
        if any(len(d) != len(data[0]) for d in data[1:]):
            plot_data = np.array(data, dtype=object)
    
    ax.boxplot(plot_data,
               boxprops=dict(color='blue'),
               whiskerprops=dict(color='blue'),
               capprops=dict(color='black'),
               medianprops=dict(color='red'),
               flierprops=dict(marker='+', markeredgecolor='blue'))
    return

# Custom ScalarFormatter to control the number of decimals
class CustomScalarFormatter(ScalarFormatter):
    def __init__(self, decimals=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = decimals

    def _set_format(self, dum1=None, dum2=None):
        if self._scientific:
            self.format = f'%.{self.decimals}e'
        else:
            self.format = f'%.{self.decimals}f'

    def __call__(self, x, pos=None):
        return self.format % x

class Profile:
    def __init__(self, stats, trace_file, concerned_stats = None):
        self.stats = [a.strip() for a in stats]
        self.trace_file = trace_file
        self.concerned_stats = concerned_stats
        self.numtrain = 0
        self.numtst = 0
        self.isolved = 0
        self.timeouts = 0
        self.abnormals = 0
        self.trainind = []
        self.tstind = []
        self.all_loc = []
        self.all_stats = {}
        self.nrows = 3
        self.ncols = 3
        self.xsize = 10
        self.ysize = 9
        self.fontsize_title = 12
        self.fontsize_xticks = 10
        self.max_shift = 1e4

    def parse(self):
        f = open(self.trace_file)
        lf = f.read()
        f.close()
        lf2 = lf.split('\n')

        # Begin to read the trace file only train and tst sets
        for i in range(len(lf2) - 1):
            lf3 = lf2[i].split(',')
            setloc = self.stats.index("SET")
            if (is_num(lf3[setloc])):
                if (int(lf3[setloc]) == 0):
                    self.numtrain = self.numtrain + 1
                    self.trainind.append(i)
                if (int(lf3[setloc]) > 0):
                    self.numtst = self.numtst + 1
                    self.tstind.append(i)

        ## This section of code extracts the relevant information
        for c_s in (self.concerned_stats if self.concerned_stats is not None else self.stats):
            self.all_stats[c_s] = []
        for i in self.trainind:
            lf3 = [a.strip() for a in lf2[i].split(',')]
            for loc in range(len(self.stats)):
                try:
                    lf3[loc] = float(lf3[loc])
                except:
                    if show_invalid_value:
                        print("Invalid value for {} - {}: {}".format(loc, self.stats[loc], lf3[loc]))
                    lf3[loc] = None
            if (lf3[self.stats.index("AlamoStatus")] == ALAMO_SUCCESS):
                self.isolved += 1
                for c_s in (self.concerned_stats if self.concerned_stats is not None else self.stats):
                    if c_s == "Size":
                        self.all_stats[c_s].append(lf3[self.stats.index("ModelSize")]/ lf3[self.stats.index("nBas")])
                    elif c_s in ["SSE", "R2", "RIC", "SSEp", "MSE", "RMSE"]:
                        self.all_stats[c_s].append(lf3[self.stats.index(c_s)])
                    elif c_s in ["BIC", "Metric1Lasso", "Metric2Lasso", "AICc", "HQC", "Cp"]:
                        if lf3[self.stats.index(c_s)] is not None:
                            self.all_stats[c_s].append(lf3[self.stats.index(c_s)])
            elif (lf3[self.stats.index("AlamoStatus")] == ALAMO_TIMEOUT):
                self.timeouts += 1
            else:
                self.abnormals += 1
            for c_s in ["TotalTime", "OLRTime", "MIPTime", "SimData"]:
                self.all_stats[c_s].append(lf3[self.stats.index(c_s)])

        if (self.isolved > 0):
            self.all_stats["Solved"].append(float(self.isolved) / float(self.numtrain))

        for i in self.tstind:
            lf3 = [a.strip() for a in lf2[i].split(',')]
            for loc in range(len(self.stats)):
                try:
                    lf3[loc] = float(lf3[loc])
                except:
                    if show_invalid_value:
                        print("Invalid value for {} - {}: {}".format(loc, self.stats[loc], lf3[loc]))
                    lf3[loc] = None
            if (lf3[self.stats.index("AlamoStatus")] == ALAMO_SUCCESS):
                for c_s in ["SSE", "R2", "RMSE"]:
                    self.all_stats[c_s + "tst"].append(lf3[self.stats.index(c_s)])

    def plot(self, fig_titles, fig_columns, fig_xticks):
        fig = plt.figure()
        fig.set_size_inches(self.xsize, self.ysize)
        this_fig_xticks = []

        axs = []
        bps = []
        datas = []
        shifts = []
        max_values = []
        min_values = []
        for i in range(len(fig_columns)):
            data = []
            xticks = []
            is_empty = True
            min_value = 0.0
            max_value = 0.0
            for k in fig_columns[i]:
                kk = fig_columns[i].index(k)
                if len(self.all_stats[k]) > 0:
                    data.append(self.all_stats[k])
                    xticks.append(fig_xticks[i][kk])
                    is_empty = False
                    min_value = min(min_value, np.min(self.all_stats[k]))
                    max_value = max(max_value, np.max(self.all_stats[k]))
            if is_empty:
                axs.append(None)
                this_fig_xticks.append(None)
                datas.append(None)
                shifts.append(None)
                min_values.append(None)
                max_values.append(None)
                continue
            else:
                axs.append(fig.add_subplot(self.nrows, self.ncols, i + 1))
            this_fig_xticks.append(xticks)
            # if min_value < 0.0 and abs(min_value) < self.max_shift:
            #     # shift a bit more to make sure the box plot does not touch boundaries
            #     min_value -= 10.0
            #     for k in range(len(data)):
            #         data[k] = [a-min_value for a in data[k]]
            datas.append(data)
            shifts.append(min_value)
            min_values.append(min_value)
            max_values.append(max_value)
            bp = stylized_boxplot(data, axs[i])

        for i in range(len(axs)):
            ax = axs[i]
            if ax is None:
                print("Sub-figure {} is omitted due to no valid columns".format(i))
                continue
            xticknames = plt.setp(ax, xticklabels=this_fig_xticks[i])
            plt.setp(xticknames, fontsize=self.fontsize_xticks)
#            plt.tight_layout()
            ax.set_title(fig_titles[i], fontsize=self.fontsize_title)
            # if shifts[i] < 0.0 and abs(shifts[i]) < self.max_shift:
            #     ax.set_title("{} (shift = {})".format(fig_titles[i], np.round(-shifts[i], 2)), fontsize=self.fontsize_title)

            # Set up the locator and format of y-axis
            delta = np.percentile([abs(a) for a in datas[i][0]], 80)
            # print(delta, min_values[i])
            # ax.set_yscale('symlog')
            # formatter = CustomScalarFormatter(decimals=1, useMathText=True)
            # formatter.set_powerlimits((-4, 4))  # Use scientific notation only for very small/large numbers
            # if max(abs(max_values[i]), abs(min_values[i])) < 1e-1 or max(abs(max_values[i]), abs(min_values[i])) > 1e3:
            #     formatter.set_scientific(True)
            # else:
            #     formatter.set_scientific(False)
            # ax.yaxis.set_major_formatter(formatter)
            # symlog_locator = SymmetricalLogLocator(base=10, linthresh=(delta if delta >= 1 else 1000))
            # symlog_locator.set_params(numticks=4)
            # symlog_locator.view_limits(min_values[i], max_values[i])
            # ax.yaxis.set_major_locator(symlog_locator)
            # ax.set_ylim(min_values[i]-10, max_values[i]+10)

        overall_stats_text = ""
        overall_stats_text += "Solved: {} out of {} instances \n".format(self.isolved, self.numtrain)
        overall_stats_text += "-     Normal termination: {} \n".format(self.isolved)
        overall_stats_text += "-     Timeouts: {} \n".format(self.timeouts)
        overall_stats_text += "-     Abnormals: {}".format(self.abnormals)
        fig.text(0.4, 0.2, overall_stats_text, fontsize=12, bbox=dict(facecolor='none', alpha=0.5))

        # Name the image
        fname = self.trace_file.split('.')[0] + '.png'
        # plt.tight_layout()
        fig.subplots_adjust(wspace=.35, hspace=.3)
        fig.savefig(fname)
        fig.savefig(fname, bbox_inches='tight', pad_inches=1)

if __name__ == "__main__":
    # ALAMO trace file reader and plotter of performance measures
    # each line of the trace file is expected to contain the following information:
    almstr = ("filename, NINPUTS, NOUTPUTS, INITIALPOINTS, OUTPUT, SET, INITIALIZER, SAMPLER, MODELER, BUILDER, "
              "GREEDYBUILD, BACKSTEPPER, GREEDYBACK, REGULARIZER, SOLVEMIP, SSEOLR, SSE, RMSE, R2, ModelSize, "
              "BIC, RIC, Cp, AICc, HQC, MSE, SSEp, MADp, OLRTime, numOLRs, OLRoneCalls, OLRoneFails, OLRgsiCalls, "
              "OLRgsiFails, OLRdgelCalls, OLRdgelFails, OLRclrCalls, OLRclrFails, OLRgmsCalls, OLRgmsFails, "
              "CLRTime, numCLRs, MIPTime, NumMIPs, LassoTime, Metric1Lasso, Metric2Lasso, LassoSuccess, LassoRed, "
              "nBasInitAct, nBas, SimTime, SimData, TotData, NdataConv, OtherTime, NumIters, IterConv, TimeConv, "
              "Step0Time, Step1Time, Step2Time, TotalTime, AlamoStatus, AlamoVersion, Model")

    # Remember to define special calculation logics for concerned statistics that do not appear in original stats
    concerned_stats = ["Size", "Solved", "R2tst", "SSEtst", "RMSEtst", "SSE", "R2", "BIC", "Metric1Lasso", "Metric2Lasso", "AICc", "HQC", "Cp", "RIC", "SSEp",
                       "MSE", "RMSE", "TotalTime", "OLRTime", "MIPTime", "SimData"]

    # Define titles, columns, xticks for each sub-figure
    fig_titles = ['Number of sampled points', 'CPU time (s)', 'R2s', 'Metrics-1', 'Metrics-2', 'Metrics-3',  'Bases Fraction']
    fig_columns = [["SimData"], ["TotalTime", "OLRTime", "MIPTime"], ["R2", "R2tst"],
                   ["BIC", "Metric1Lasso", "Metric2Lasso", "AICc"], ["RIC", "SSEp", "HQC", "Cp"],
                   ["MSE", "RMSE", "SSE", "SSEtst"],  ["Size"]]

    fig_xticks = [["Samples"], ["Total", "OLR", "MIP"], ["R2", "R2tst"],
                   ["BIC", "Las1", "Las2", "AICc"], ["RIC", "SSEp", "HQC", "Cp"],
                   ["MSE", "RMSE", "SSE", "SSEtst"], ["Basis"]]

    script, infile = argv

    ppf = Profile(almstr.split(','), infile, concerned_stats=concerned_stats)
    ppf.parse()
    ppf.plot(fig_titles, fig_columns, fig_xticks)
