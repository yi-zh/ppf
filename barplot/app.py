'''
Script for performance profiles adapted in many ways. Originally written by
Carlos, the script was later modified by Anatoliy and Yi.
'''
from __future__ import print_function, division
import argparse
import ntpath
import math
from matplotlib.ticker import ScalarFormatter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import re
import tempfile
from operator import itemgetter

'''
Example: python app.py trace1 trace2 --failtime 3600 -fe -fh -tf
'''

def get_name(name):
    '''
    Gets name of the instance by removing flags like *U, *P that maybe added
    by examiner to indicate issues with the solution
    '''
    return re.sub(r'\*[A-Z]-', '', re.sub(r'\*[a-z]-', '', name))

def write_values(solver, data_file, time_factor):
    '''
    Writes the values in a table format that can be used by gnuplot for
    making prformance profiles
    '''
    nprob = len(solver.vals)
    if time_factor:
        vals = solver.ratios
    else:
        vals = solver.vals

    min_val = min(vals)
    cnt = 0
    for k in range(nprob):
        if np.isclose(vals[k], min_val):
            cnt += 1
        else:
            break

    data_file.write("{0} {1}\n".format(min_val, 1.0*cnt/nprob))
    for k in range(cnt, nprob):
        data_file.write("{0} {1}\n".format(vals[k], 1.0*(k+1)/nprob))
    return


def display_table(solvers, rat_only):
    '''
    Display table of performance metrics on screen
    '''
    nprob = len(solvers[0].vals)
    for j in range(nprob):
        print("%4d/%d" % (j+1, nprob))
        if rat_only:
            for solv in solvers:
                print("%8.2f" % solv.vals[j])
        for solv in solvers:
            print("%8.2f" % solv.ratios[i])
        print("")
    return


def find_terminal_type(args):
    '''
    Find the text for terminal type based on args
    '''
    if args.eps:
        terminal_type = "postscript eps"
        extension = "eps"
        if args.b_and_w:
            colors = "monochrome enhanced dashed"
        else:
            colors = "color dashed"

    else:
        terminal_type = "png"
        extension = "png"
        colors = ""

    return (terminal_type, extension, colors)


def gnu_plot(solvers, args):
    '''
    Plot the performance profiles using GNUPlot
    '''
    data_files = [tempfile.NamedTemporaryFile(delete=False, mode='w')
                  for _ in range(len(solvers))]
    for k, solver in enumerate(solvers):
        write_values(solver, data_files[k], args.time_factor)
        data_files[k].close()

    terminal_type, extension, colors = find_terminal_type(args)
    gnu_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
    gnu_file.write("set terminal {0} {1}\n".format(terminal_type, colors))
    gnu_file.write("set output \"{0}.{1}\"\n".format(args.filename, extension))

    gnu_file.write('set title "%s"\n' % args.title)
    gnu_file.write("set key right bottom\n")
    if args.time_factor:
        gnu_file.write('set xlabel "Time factor"\n')
    else:
        gnu_file.write('set xlabel "CPU time"\n')

    #    gnu_file.write('set xlabel "not more than x-times slower than '
    #                   'best solver"\n')
    gnu_file.write('set ylabel "proportion of problems solved"\n')
    gnu_file.write("set log x\n")
    gnu_file.write("set logscale x %f\n" % args.log_base)
    gnu_file.write("set yrange [%f:%f]\n" % (args.ylo, args.yup))
    gnu_file.write("set ytics 0.1\n")
    if args.xlo > 1e-10:
        gnu_file.write("set xrange [%f:%f]\n" % (args.xlo, args.xup))
    else:
        gnu_file.write("set xrange [:%f]\n" % args.xup)
    if ARGS.colors:
        for j in range(len(solvers)):
            gnu_file.write("set style line {0} lt rgb \"{1}\" "
                           "lw 2\n".format(j+1, ARGS.colors[j]))

    for j, solv in enumerate(solvers):
        if j == 0:
            gnu_file.write("plot ")
        else:
            gnu_file.write(", ")
        if ARGS.colors:
            gnu_file.write("\"%s\" using 1:2 with steps ls %d title "
                           "\"%s\"" % (data_files[j].name, j+1, solv.name))
        else:
            gnu_file.write("\"%s\" using 1:2 with steps lw 2 title "
                           "\"%s\"" % (data_files[j].name, solv.name))
    gnu_file.close()
    os.system("gnuplot {0}".format(gnu_file.name))
    return


def validate_solvers(solvers, args, instances):
    '''
    Validates solutions by solvers to check if:
        a. They are within tolerances
        b. Global optimum claimed by one solver is not beaten by an examiner
        verified solution from other solver
        c. If the lower bounds from one solver violate best known solution
        Note: Not used for now
    '''
    for prob in instances:
        metrics = [solv.metrics[prob] for solv in solvers if prob in
                   solv.metrics and solv.metrics[prob] is not None]
        # This would be the best known solution since Examiner has verified it
        if not metrics:
            continue
        best_sol = min(metrics, key=itemgetter(0))[1]
        for solv in solvers:
            stats = solv.metrics.get(prob)
            if stats is None:
                continue
            if stats[0] > best_sol + 1e-6:
                # Did not find the best solution.
                solv.metrics[prob] = None
                continue

            if stats[1] > best_sol + 1e-6:
                # Lower bound from solution violates best known
                # Replace as failure
                solv.metrics[prob] = None
                continue

            if stats[2] < args.max_val - 1:
                # Since solver terminated before time limit, make sure
                # tolerances are satisfied
                gap = abs(stats[0] - stats[1])
                # Tolerances violated. Failure
                # Use definitions of all, optca, epsr and optcr

                failure = (gap > args.epsa and gap > args.epsr*abs(stats[1])
                           and gap > args.epsr*max(abs(stats[0]),
                                                   abs(stats[1])))
                if failure:
                    solv.metrics[prob] = None
    return


def read_best_known(file_name):
    '''
    Reads file with best_known solutions available for a library
    '''
    headers = ['filename', 'direction', 'status', 'obj']
    best_known = {}
    with open(file_name, 'r') as read_file:
        reader = csv.DictReader(read_file, fieldnames=headers, delimiter=" ")
        for row in reader:
            prob = row["filename"]
            try:
                bfs = float(row["obj"])
            except ValueError:
                continue
            if int(row["direction"]) == 1:
                bfs = -bfs

            if prob in best_known:
                print("Please check the file: {0}. It contains duplicate"
                      " entries for problem: {1}.".format(file_name, prob))
            best_known[prob] = bfs
    return best_known


def calculate_ratios(solvers, args, instances, best_known):
    '''
    Calculate the ratios comparing solver time to the best solver for all
    problem instances
    '''
    xpressmaxtt = 0.0
    fails = {}
    for solv in solvers:
        solv.ratios = [None] * len(instances)
        solv.vals = [None] * len(instances)
        fails[solv.name] = 0
    for idx, prob in enumerate(instances):
        metrics = [solv.metrics[prob] for solv in solvers if
                   solv.metrics.get(prob) is not None and len(solv.metrics[prob]) > 0]
    
        if not metrics:
            for solv in solvers:
                solv.fails["no_metrics"] += 1
                solv.vals[idx] = args.failtime
            continue
        best_time = min(metrics, key=itemgetter(2))[2]
        best_time = min(best_time, args.max_val)
        bfs = min(metrics, key=itemgetter(0))[0]

        if best_known is not None:
            best_available = best_known.get(prob)
            if best_available:
                bfs = min(bfs, best_available)

        for solv in solvers:                
            if solv.metrics.get(prob) is not None and len(solv.metrics[prob]) > 0:
                solv_time = solv.metrics[prob][2]
                upper_bound = solv.metrics[prob][0]
                failed = False
                if args.primal:
                    if np.isclose(upper_bound, bfs, rtol=args.epsr):
                        solv_time = min(solv_time, args.max_val)
                    elif solv_time <= args.max_val and (abs(bfs - upper_bound) <= args.epsa or \
                        (abs(bfs - upper_bound) / abs(bfs) if abs(bfs) > 1e-8 else abs(bfs - upper_bound)) <= args.epsr):
                        pass
                    else:
                        fails[solv.name] += 1
                        solv.fails["terminated_but_suboptimal"] += 1
                        failed = True
                else:
                    if solv_time >= args.max_val:
                        if np.isclose(solv_time, args.max_val):
                            solv_time = args.max_val
                        fails[solv.name] += 1
                        solv.fails["timed_out"] += 1
                        failed = True
                    elif abs(bfs - upper_bound) > args.epsa and \
                        (abs(bfs - upper_bound) / abs(bfs) if abs(bfs) > 1e-8 else abs(bfs - upper_bound)) > args.epsr:
                        fails[solv.name] += 1
                        # if not np.isclose(solv_time, args.max_val) and solv.name == "BARON":
                        #     print(f"Solver {solv.name}'s solution of {upper_bound:.6f} is worse than the best solution of {bfs:.6f} for instance {prob} (solve time: {solv_time:.2f} s)")
                        # print(f"\tabsolute: {abs(bfs - upper_bound):.3e}, relative: {(abs(bfs - upper_bound) / abs(bfs) if abs(bfs) > 1e-8 else abs(bfs - upper_bound)):.3e}")
                        solv.fails["terminated_but_suboptimal"] += 1
                        failed = True
                if failed:
                    solv.failed_instances.add(prob)
                    solv.vals[idx] = args.failtime
                else:
                    solv.ratios[idx] = float(solv_time)/best_time
                    solv.vals[idx] = solv_time
                    if solv.name == "Xpress":
                        xpressmaxtt = max(xpressmaxtt, solv_time)
            else:
                solv.fails["no_metrics"] += 1
                fails[solv.name] += 1
    
    for solv in solvers:
        solv.vals = [val if val is not None else args.failtime
                     for val in solv.vals]
        solv.ratios = [val if val is not None else args.failtime
                       for val in solv.ratios]
        solv.fails["total"] = sum(solv.fails.values())
        print(f"Fails for solver {solv.name}:")
        print(solv.fails)
        if solv.name == "Xpress":
            a = np.sort(solv.vals)
            with open('logfile.txt', 'a') as log_file:
                for ele in a:
                    print("value:", ele, file=log_file)
#    print("Xpress maxtt", xpressmaxtt)
    return


def sort_arrange(values, xticks, instance_count):
    '''
    Arranges values in a cumulative manner to fit between xticks
    '''

    sorted_values = np.sort(values)
    out = []
    for x_t in xticks:
        size = len(np.where(sorted_values <= x_t)[0])
        out.append(size)

    for j, perf in enumerate(out):
        out[j] = float(perf) * 100 / instance_count

    return out

class Solver(object):
    '''
    Solver object to store data from a tracefile
    '''
    def __init__(self, file_handle, name):
        self.file = file_handle
        self.name = file_handle if name is None else name
        self.metrics = {}
        self.ratios = []
        self.vals = []
        self.data = None
        self.maxval = -1
        self.fails = {
            "no_metrics": 0,
            "terminated_but_suboptimal": 0,
            "timed_out": 0,
            "total": 0
        }
        self.failed_instances = set()
        self.all_instance_count = 0
        self.raw_data = None

    def read_trace(self, args):
        '''
        Processes the tracefile and stores all information. Only need a few
        fields for now, but store all to allow room for extensibility
        '''
        headers = ['filename', 'modeltype', 'solvername', 'NLP def', 'MIP def',
                   'juliantoday', 'direction', 'equnum', 'varnum', 'dvarnum',
                   'nz', 'nlnz', 'optfile', 'modelstatus', 'solverstatus',
                   'obj', 'objest', 'res used', 'iter used', 'dom used',
                   'nodes used']
        reader = csv.DictReader(self.file, fieldnames=headers, restkey='user1')
        # DictReader skips empty lines by default. Skip comments
        self.data = [row for row in reader if row['filename'] is not None and
                     not row['filename'].startswith(('%', '#'))]
        self.all_instance_count = len(self.data)
        self.raw_data = self.data
        self.data = [row for row in self.data if '*P-' not in row['filename'] and '*U-' not in row['filename']]
        if args.nonprimalfailures:
            self.data = [row for row in self.data if '*D-' not in row['filename']]
            self.data = [row for row in self.data if '*ECS-' not in row['filename']]
            self.data = [row for row in self.data if '*PCS-' not in row['filename']]
            self.data = [row for row in self.data if '*DCS-' not in row['filename']]
        return

    def preprocess_trace(self):
        '''
        Reproduce the preprocessing carried out by Nick's c function in bashrc
        '''
        if self.data is None:
            raise ValueError("Call read_trace before calling preprocess_trace")

        for row in self.data:
            row["filename"] = row["filename"].replace("*PCS-", "")
            row["filename"] = row["filename"].replace("*ECS-", "")
            row["filename"] = row["filename"].replace("*DCS-", "")
            row["filename"] = row["filename"].replace("*D-", "")
            row["filename"] = row["filename"].replace("*W-", "")

        return

    def calc_metrics(self, args):
        '''
        Calculates the important metrics needed from data.
        '''
        if self.data is None:
            raise ValueError("Call read_trace before calling calc_metrics")

        for row in self.data:
            # This is a failure instance. Get name and continue
            if row['filename'] == '*minlp2':
                # In this failure instance, name is stored in modeltype field
                if row['modeltype'] is not None:
                    self.metrics[get_name(row['modeltype'])] = None
                continue
            if row['filename'] is None:
                continue

            name = get_name(row['filename'])
            self.metrics[name] = []
            
            if name.startswith('*'):
                continue

            solver_status = int(row["solverstatus"])
            model_status = int(row["modelstatus"])

            # Check http://www.gamsworld.org/performance/status_codes.htm
            # for status codes that have been excluded
            if model_status in (3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19):
                continue

            if solver_status in (5, 6, 7, 9, 10, 11, 12, 13):
                continue

            try:
                upper_bound = float(row['obj'])
            except ValueError:
                upper_bound = 1e51

            try:
                lower_bound = float(row['objest'])
            except ValueError:
                lower_bound = -1e51

            try:
                time = float(row['res used'])
            except ValueError:
                 # Can really do nothing if time is not available
                self.metrics[name] = None
                continue

#            if time >= args.failtime - 1e-4:
#                continue

            if abs(upper_bound - lower_bound) > args.epsa and time < args.failtime - 1e-4:
                continue

            self.metrics[name] = [upper_bound, lower_bound, time]

            if row['direction'] == '1':
                # Maximization problem. Convert to min for consistency
                self.metrics[name][0] *= -1
                self.metrics[name][1] *= -1
            self.metrics[name][2] = max(self.metrics[name][2], args.min_val)
            self.maxval = max(self.maxval, self.metrics[name][2])
        return

    def return_instances(self):
        '''
        Return the name of instances for which solver has data
        '''
        return set(self.metrics.keys())

    def can_solve(self, solvers, instances, args):
        '''
        Calculates metrics for the special solver can_solve, that includes
        aggregated stats for all solvers, combined based on best performance
        For the moment, only compares based on time
        '''
        for prob in instances:
            time = np.inf
            gap = np.inf
            best_lb = -np.inf
            best_ub = np.inf
            any_solver = False
            for solv in solvers:
                metrics = solv.metrics.get(prob)
                if metrics is not None and len(metrics) > 0:
                    stime = metrics[2]
                    upper_bound = metrics[0]
                    lower_bound = metrics[1]
                    if stime > 0:
                        any_solver = True
                        time = max(args.min_val, min(stime, time))
                    #    print(f"{prob} solved by {solv.name} in {time:.1f} s")

                    #if abs(upper_bound - lower_bound) < gap:
                    gap = abs(upper_bound - lower_bound)
                    best_lb = max(lower_bound, best_lb)
                    best_ub = min(upper_bound, best_ub)

            if any_solver:
                # Only interested in best time for now
                self.metrics[prob] = (best_ub, best_lb, time)
            else:
                self.metrics[prob] = None
        return

def collect_all_instances(solver):
    rows = solver.raw_data
    for row in rows:
        row["filename"] = row["filename"].replace("*PCS-", "")
        row["filename"] = row["filename"].replace("*ECS-", "")
        row["filename"] = row["filename"].replace("*DCS-", "")
        row["filename"] = row["filename"].replace("*D-", "")
        row["filename"] = row["filename"].replace("*W-", "")
        row["filename"] = row["filename"].replace("*P-", "")
        row["filename"] = row["filename"].replace("*U-", "")

    return [row["filename"] for row in rows]

def gap_closed(lower_bound: float, upper_bound:float, epsa: float, epsr: float) -> bool:
    absolute_gap_closed = (upper_bound - lower_bound <= epsa)
    if np.equal(lower_bound, 0):
        relative_gap_closed = (upper_bound <= epsr)
    else:
        relative_gap_closed = ((upper_bound - lower_bound) / abs(lower_bound) <= epsr)
    return absolute_gap_closed or relative_gap_closed

def setup_argument_parser():
    '''
    Sets up the parser object
    '''
    parser = argparse.ArgumentParser(description='Generate pretty performance'
                                                 ' profiles from the traces '
                                                 'provided for solvers')
    parser.add_argument("traces", nargs='+', help='Path to all tracefiles'
                        'separated by a space', type=argparse.FileType('r'))
    parser.add_argument("--names", nargs='+', help='Names to be used in'
                        ' legend. Default is to use file name')
    parser.add_argument("--version", action="version", version='%(prog)s 1.0')
    parser.add_argument("-tf", "--time-factor", action='store_true', default=False,
                        help='Create profile with absolute values for the '
                             'metric. Uses relative by default')
    parser.add_argument("-s", "--solve", action='store_true',
                        default=False, help="Add curve for CAN_SOLVE")
    parser.add_argument("-nop", "--no-preprocess", action='store_true',
                        default=False, help="Do not preprocess out EXAMINER"
                        "flags")
    parser.add_argument("-b", "--b-and-w", action='store_true', default=False,
                        help="Create profile in black and white")
    parser.add_argument("-n", "--no-gnu", action='store_true', default=False,
                        help="Do not create EPS with GNUplot. A table output "
                             "is printed on screen")
    parser.add_argument("-l", "--log-base", action='store', type=int,
                        help="Use log with a different base for the x-axis. "
                             "(Default: 10)", default=10)
    parser.add_argument("-t", "--title", action="store", type=str,
                        help="Title for the performance profile "
                             "(Default: Performance Profile)",
                        default="Performance Profile")
    parser.add_argument("-m", "--min-val", action="store", type=float,
                        help="All metric values less than --min-val are "
                             "replaced by --min-val. (Default: 0.01)",
                        default=0.01)
    parser.add_argument("-M", "--max-val", action="store", type=float,
                        help="All metric values greater than --max-val are "
                             "considered failures. Set to timelimit used in "
                             "tests (Default: 500)",
                        default=500)
    parser.add_argument("-x", "--xlo", action="store", type=float,
                        help="Lower bound for X-axis. (Default: 0)", default=0)
    parser.add_argument("-X", "--xup", action="store", type=float,
                        help="Upper bound for X-axis. (Default: 1e50)",
                        default=1e50)
    parser.add_argument("-y", "--ylo", action="store", type=float,
                        help="Lower bound for Y-axis. (Default: 0)", default=0)
    parser.add_argument("-Y", "--yup", action="store", type=float,
                        help="Upper bound for Y-axis. (Default: 100)", default=100)
    parser.add_argument("-E", "--epsa", action="store", type=float,
                        help="Absolute tolerance used in tests. "
                             "(Default: 1e-5)", default=1e-5)
    parser.add_argument("-R", "--epsr", action="store", type=float,
                        help="Absolute tolerance used in tests. "
                             "(Default: 1e-5)", default=1e-5)
    parser.add_argument("--failtime", action='store', default=3600, type=float,
                        help='Time value assigned to failures. (Default: '
                        '3600)')
    parser.add_argument("--shift", action='store', default=10, type=float,
                        help='Shift for geometric mean in seconds. (Default: '
                        '10)')
    parser.add_argument("--failgap", action='store', default=1e20, type=float,
                        help='Gap value assigned to failures. (Default: '
                        '1e20)')
    parser.add_argument("--timefilename", action="store", type=str,
                        help="Name for file where time plot is stored. "
                             "(Default: times.png)",
                        default="times.png")
    parser.add_argument("--gapfilename", action="store", type=str,
                        help="Name for file where gap plot is stored. "
                             "(Default: gaps.png)",
                        default="gaps.png")
    parser.add_argument("--best-sol-file", action="store", type=str,
                        help="File contains currently best known solutions "
                        "for problems in testlib", default=None)
    parser.add_argument("--instancesize", action='store_true',
                        help="Print instance size statistics"
                             "(Default: False)"),
    parser.add_argument("--size_level", action='store', type=int,
                        help="Level of box plot for model size: 0 - only show All and Solvable (-s), "
                             "1 - show all solvers"
                             "(Default: 0)", default=0)
    parser.add_argument("--colors", action="store", type=str,
                        help="Colors for plot, in same order as solvers. r: red, g: green, b: blue, k: black"
                             "(Example: rgbkrgb)"),
    parser.add_argument("--linestyles", action="store", type=str,
                        help="Line styles for plot, in same order as solvers. | is solid, - is dashed, : is dotted."
                             "(Example: '|||---:')"),
    parser.add_argument("-p", "--primal", action='store_true',
                        default=False, help="Generate plot based on primal bounds.")
    parser.add_argument("-g", "--gap", action='store_true',
                        default=False, help="Generate plot of remaining gaps.")
    parser.add_argument("--nonprimalfailures", action='store_true',
                        default=False, help="Count non-primal infeasibilities (*D-, *ECS-, *PCS-, *DCS-) as failures. Defaults to False.")
    parser.add_argument("-minf", "--min-val-filter", action="store", type=float,
                        help="All metric values smaller than --min-val-filter are "
                             "considered trivial.",
                        default=1)
    parser.add_argument("-maxf", "--max-val-filter", action="store", type=float,
                        help="All metric values larger than --max-val-filter are "
                             "considered time-out.",
                        default=3600)
    parser.add_argument("-fe", "--filter-easy", action='store_true',
                        default=False, help="Filter out trivial instances.")
    parser.add_argument("-fh", "--filter-hard", action='store_true',
                        default=False, help="Filter out time-out instances.")
    return parser


def calculate_gaps(solvers, args, instances, best_known):
    '''
    Calculate the gaps remaining for every problem instance not solved
    '''

    for solv in solvers:
        solv.gaps = [args.failgap]*len(instances)

    for idx, prob in enumerate(instances):
        bfs = min([solv.metrics.get(prob)[0] if solv.metrics.get(prob)
                   else 1e51 for solv in solvers])
        if best_known is not None:
            best_available = best_known.get(prob)
            if best_available:
                bfs = min(bfs, best_available)

        for solv in solvers:
            if solv.metrics.get(prob):
                upper_bound = solv.metrics[prob][0]
                lower_bound = solv.metrics[prob][1]
                solv_time = solv.metrics[prob][2]
                terminated = (solv_time < args.max_val and not
                              np.isclose(solv_time, args.max_val))
                if terminated and (abs(bfs - upper_bound) > args.epsa):
                    solv.gaps[idx] = args.failgap
                elif gap_closed(lower_bound, upper_bound, epsa=args.epsa,
                                epsr=args.epsr):
                    solv.gaps[idx] = 0

                else:
                    if np.isneginf(lower_bound) or np.isinf(upper_bound):
                        solv.gaps[idx] = args.failgap
                    else:
                        solv.gaps[idx] = (upper_bound - lower_bound)/(
                            max(abs(lower_bound), 0.001))*100


def calculate_plot_values(solvers, args, instance_count):
    '''
    Calculate the actual x, y values to be plotted
    '''
    if args.time_factor:
        time = [] + (np.linspace(1, 10.0, num=100)).tolist()
        time += (np.linspace(20, 100, num=9)).tolist()
        time += (np.linspace(105, 1000, num=int((1000 - 100) // 5))).tolist()
    else:
        time = [] + (np.linspace(1, 10.0, num=10)).tolist()
        time += (np.linspace(20, 100, num=9)).tolist()
        time += (np.linspace(105, args.failtime, num=int((args.failtime - 100) // 5))).tolist()

    gap = [] + (np.linspace(0.0, 100.0, num=101)).tolist()
    rho_time = [None for _ in solvers]
    rho_gap = [None for _ in solvers]
    for idx, solv in enumerate(solvers):
        if args.time_factor:
            rho_time[idx] = sort_arrange([v for v in solv.ratios if not np.isclose(v, args.failtime) and v is not None], time, instance_count)
        else:
            rho_time[idx] = sort_arrange([v for v in solv.vals if not np.isclose(v, args.failtime) and v is not None], time, instance_count)
        if args.gap:
            rho_gap[idx] = sort_arrange(solv.gaps, gap, instance_count)
    return time, rho_time, gap, rho_gap

def stylized_boxplot(data, ax=None):
    if ax is None:
        ax = plt.gca()
    plot_data = data
    if len(data) > 1:
        if any(len(d) != len(data[0]) for d in data[1:]):
            plot_data = np.array(data, dtype=object)
    ax.boxplot(plot_data,
               # showfliers = False,
               boxprops=dict(color='blue'),
               whiskerprops=dict(color='blue'),
               capprops=dict(color='black'),
               medianprops=dict(color='red'),
               flierprops=dict(marker='+', markeredgecolor='blue'))
    return

def instance_size_plot(ARGS, all_stats, SOLVERS):
    fig = plt.figure()
    nrows = 3
    ncols = 2
    fontsize_title = 12
    fontsize_xticks = 10
    this_fig_xticks = []

    axs = []
    datas = []
    fig_columns = []
    fig_xticks = []
    fig_titles = ['Number of problems', 'Number of constraints', 'Number of variables', 'Number of discrete variables', 'Number of nonzero elements', 'Number of nonlinear entries']
    fig_keywords = ['nprob', 'ncons', 'nvar', 'nintvar', 'nz', 'nlnz']
    k = 1
    for i in range(len(fig_keywords)):
        columns = []
        xticks = []
        columns.append("ALL_{}".format(fig_keywords[i]))
        xticks.append("All")
        for solver in SOLVERS:
            if ARGS.size_level == 0 and solver.name != "Solvable":
                continue
            if i == 0:
                k += 1
            columns.append("{}_{}".format(solver.name, fig_keywords[i]))
            xticks.append(solver.name)
        fig_columns.append(columns)
        fig_xticks.append(xticks)
    fig.set_size_inches(3*(k+1), 9)
    for i in range(len(fig_columns)):
        data = []
        xticks = []
        is_empty = True
        for k in fig_columns[i]:
            kk = fig_columns[i].index(k)
            if len(all_stats[k]) > 0:
                data.append(all_stats[k])
                xticks.append(fig_xticks[i][kk])
                is_empty = False
            else:
                data.append([0])
                xticks.append(fig_xticks[i][kk])
                is_empty = False
        if is_empty:
            axs.append(None)
            this_fig_xticks.append(None)
            datas.append(None)
            continue
        else:
            axs.append(fig.add_subplot(nrows, ncols, i + 1))
        this_fig_xticks.append(xticks)
        datas.append(data)
        stylized_boxplot(data, axs[i])

    for i in range(len(axs)):
        ax = axs[i]
        if ax is None:
            print("Sub-figure {} is omitted due to no valid columns".format(i))
            continue
        xticknames = plt.setp(ax, xticklabels=this_fig_xticks[i])
        plt.setp(xticknames, fontsize=fontsize_xticks)
        ax.set_title(fig_titles[i], fontsize=fontsize_title)

    # Name the image
    fname = 'size.png'
    # plt.tight_layout()
    fig.subplots_adjust(wspace=.35, hspace=.3)
    fig.savefig(fname)
    fig.savefig(fname, bbox_inches='tight', pad_inches=1)

def count_model_size(solver):
    variable_counts = {}
    int_variable_counts = {}
    constraint_counts = {}
    nz_counts = {}
    nonlinear_nz_counts = {}
    # baron_trace_filename = next(f.name for f in traces if "BARON" in f.name)
    # with open(baron_trace_filename, "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    # for line in lines:
    #     tokens = line.split(",")
    #     instance_name = tokens[22]
    #     constraint_count = int(tokens[23])
    #     variable_count = int(tokens[24])
    #     if constraint_count < 0 or variable_count < 0:
    #         print(f"Could not read number of variables and constraints in instance {instance_name}")
    #         continue
    #     constraint_counts[instance_name] = constraint_count
    #     variable_counts[instance_name] = variable_count

    rows = solver.raw_data
    for row in rows:
        row["filename"] = row["filename"].replace("*PCS-", "")
        row["filename"] = row["filename"].replace("*ECS-", "")
        row["filename"] = row["filename"].replace("*DCS-", "")
        row["filename"] = row["filename"].replace("*D-", "")
        row["filename"] = row["filename"].replace("*W-", "")
        row["filename"] = row["filename"].replace("*P-", "")
        row["filename"] = row["filename"].replace("*U-", "")
        variable_counts[row["filename"]] = int(row["varnum"])
        int_variable_counts[row["filename"]] = int(row["dvarnum"])
        constraint_counts[row["filename"]] = int(row["equnum"])
        nz_counts[row["filename"]] = int(row["nz"])
        nonlinear_nz_counts[row["filename"]] = int(row["nlnz"])
    return variable_counts, int_variable_counts, constraint_counts, nz_counts, nonlinear_nz_counts

def filter_instances(solvers, instances, args):
    '''
    Filter out trivial and time-out instances
    '''
    dropInstances = []
    if not args.filter_easy and not args.filter_hard:
        return dropInstances
    for prob in instances:
        mintime = np.inf
        maxtime = 0
        time = np.inf
        any_solver = False
        for solv in solvers:
            metrics = solv.metrics.get(prob)
            if metrics is not None and len(metrics) > 0:
                stime = metrics[2]
                if stime > 0:
                    any_solver = True
                    mintime = min(mintime, stime)
                    maxtime = max(maxtime, stime)
                    time = max(0.0, min(stime, time))
        if any_solver and maxtime > args.min_val_filter and mintime < args.max_val_filter:
            continue
        else:
            if (args.filter_easy and maxtime <= args.min_val_filter) or (args.filter_hard and mintime >= args.max_val_filter):
                dropInstances.append(prob)

    print("Filter out {} out of {} instances".format(len(dropInstances), len(instances)))
    return dropInstances


if __name__ == "__main__":
    PARSER = setup_argument_parser()
    ARGS = PARSER.parse_args()
    if ARGS.names is None:
        # Use filenames as names
        ARGS.names = [ntpath.basename(f.name) for f in ARGS.traces]

    if len(ARGS.names) != len(ARGS.traces):
        raise ValueError("Number of names for legend does not match number of"
                         "traces")

    NTRACES = len(ARGS.traces)
    SOLVERS = [Solver(ARGS.traces[i], ARGS.names[i]) for i in range(NTRACES)]
    for i in range(NTRACES):
        SOLVERS[i].read_trace(ARGS)
        if not ARGS.no_preprocess:
            SOLVERS[i].preprocess_trace()
        SOLVERS[i].calc_metrics(ARGS)

    # INSTANCES = set()
    for i, s in enumerate(SOLVERS):
        for j in range(i):
            if s.all_instance_count != SOLVERS[j].all_instance_count:
                print(f"Warning: trace file {SOLVERS[j].file.name} includes {SOLVERS[j].all_instance_count} instances, but trace file {s.file.name} includes {s.all_instance_count} instances")

    INSTANCES = collect_all_instances(SOLVERS[0])
    INSTANCES.sort()

    if ARGS.solve:
        CAN_SOLVE = Solver(None, "Solvable")
        CAN_SOLVE.can_solve(SOLVERS, INSTANCES, ARGS)
        SOLVERS.append(CAN_SOLVE)
        ARGS.names.append("Solvable")

    DROPINSTANCES = filter_instances(SOLVERS, INSTANCES, ARGS)

    INSTANCES = [x for x in INSTANCES if x not in DROPINSTANCES]

    if ARGS.best_sol_file:
        BEST_KNOWN = read_best_known(ARGS.best_sol_file)
    else:
        BEST_KNOWN = None

    ARGS.max_val = ARGS.failtime
    calculate_ratios(SOLVERS, ARGS, INSTANCES, BEST_KNOWN)
    if ARGS.gap:
        calculate_gaps(SOLVERS, ARGS, INSTANCES, BEST_KNOWN)
    (TIME, RHO_TIME, GAP, RHO_GAP) = calculate_plot_values(SOLVERS, ARGS, len(INSTANCES))
    LABELS_SIZE = 26
    TITLE_SIZE = 26
    TICKS_SIZE = 24
    LEGEND_SIZE = 20
    plt.rcParams["font.family"] = "serif"

    if not ARGS.colors and not ARGS.linestyles:
        i = 0
        toggle = 0
        COLORS = ""
        STYLES = []
        while i < len(SOLVERS):
            COLORS = COLORS + "rbg"
            if toggle == 0:
                STYLES.extend(["solid", "solid", "solid"])
            elif toggle == 1:
                STYLES.extend(["dashed", "dashed", "dashed"])
            else:
                STYLES.extend(["dotted", "dotted", "dotted"])
            toggle = (toggle + 1) % 3
            i += 3
    elif (ARGS.colors and not ARGS.linestyles) or (not ARGS.colors and ARGS.linestyles):
        exit("Error: must specify both colors and styles, or not specify either")
    else:
        if any(c not in "rgbk" for c in ARGS.colors):
            exit("Invalid color. Colors must be in ['r', 'g', 'b', 'k'].")
        if any(s not in "|:-" for s in ARGS.linestyles):
            exit("Invalid line style. Line styles must be in ['|', ':', '-'].")
        if len(ARGS.colors) == len(SOLVERS) - 1 and len(ARGS.linestyles) == len(SOLVERS) - 1 and ARGS.solve:
            exit("Error: please specify a color and line style for Solvable.")
        if len(ARGS.colors) != len(ARGS.linestyles) or len(ARGS.colors) != len(SOLVERS) or len(ARGS.linestyles) != len(SOLVERS):
            exit("Error: mismatch between number of colors, line styles, and solvers.")
        COLORS = [c for c in ARGS.colors]
        line_styles = {'|': 'solid', '-': 'dashed', ':': 'dotted'}
        STYLES = [line_styles[s] for s in ARGS.linestyles]

    geometric_means = {}
    max_solved_instance_count = 0
    for i, solver in enumerate(SOLVERS):
        times = np.array([min(v, ARGS.failtime) + ARGS.shift if v is not None else ARGS.failtime + ARGS.shift for v in solver.vals])
        geo_mean = np.exp(np.log(times).mean()) - ARGS.shift
        geometric_means[solver.name] = geo_mean
        solved_instance_count = len(INSTANCES) - solver.fails["total"]
        if ARGS.primal:
            solve_message = "found the best solution for"
        else:
            solve_message = "solved"
        max_solved_instance_count = max(max_solved_instance_count, solved_instance_count)
        print(f"{solver.name} {solve_message} {solved_instance_count} instances ({solved_instance_count * 100 / len(INSTANCES):.1f}% of {len(INSTANCES):d}) with a geometric mean of {geo_mean:.1f} s")
    
    # Computes instance size statistics. Enable with --instancesize
    if ARGS.instancesize:
        variable_counts, int_variable_counts, constraint_counts, nz_counts, nonlinear_nz_counts = count_model_size(SOLVERS[0])
        average_variable_count = np.mean([variable_counts[instance] for instance in variable_counts])
        average_int_variable_count = np.mean([int_variable_counts[instance] for instance in int_variable_counts])
        average_constraint_count = np.mean(np.array([constraint_counts[instance] for instance in constraint_counts]))
        average_nz_count = np.mean(np.array([nz_counts[instance] for instance in nz_counts]))
        average_nlnz_count = np.mean(np.array([nonlinear_nz_counts[instance] for instance in nonlinear_nz_counts]))
        solved_instance_var_counts = {}
        solved_instance_int_var_counts = {}
        solved_instance_cons_counts = {}
        solved_instance_nz_counts = {}
        solved_instance_nlnz_counts = {}
        for solver in SOLVERS:
            solved_instance_var_counts[solver] = []
            solved_instance_int_var_counts[solver] = []
            solved_instance_cons_counts[solver] = []
            solved_instance_nz_counts[solver] = []
            solved_instance_nlnz_counts[solver] = []
            for i, instance in enumerate(INSTANCES):
                if solver.vals[i] < ARGS.failtime - 1:
                    solved_instance_var_counts[solver].append(variable_counts[instance])
                    solved_instance_int_var_counts[solver].append(int_variable_counts[instance])
                    solved_instance_cons_counts[solver].append(constraint_counts[instance])
                    solved_instance_nz_counts[solver].append(nz_counts[instance])
                    solved_instance_nlnz_counts[solver].append(nonlinear_nz_counts[instance])
            solver_average_var_count = np.mean(np.array(solved_instance_var_counts[solver]))
            solver_average_int_variable_count = np.mean(np.array(solved_instance_int_var_counts[solver]))
            solver_average_cons_count = np.mean(np.array(solved_instance_cons_counts[solver]))
            solver_average_nz_count = np.mean(np.array(solved_instance_nz_counts[solver]))
            solver_average_nlnz_count = np.mean(np.array(solved_instance_nlnz_counts[solver]))
            print(f"Solved instances for solver {solver.name} had an average of {solver_average_var_count:.0f} variables ({100 * solver_average_var_count / average_variable_count:.1f}% of library average), {solver_average_int_variable_count:.0f} integer variables ({100 * solver_average_int_variable_count / average_int_variable_count:.1f}% of library average), {solver_average_cons_count:.0f} constraints ({100 * solver_average_cons_count / average_constraint_count:.1f}% of library average)")
        print(f"Library stats          : mean || min, 1st-quantile, median, 3rd-quantile, max):")
        stats = [variable_counts[instance] for instance in variable_counts]
        print(f"Variable stats         : {np.mean(stats):.0f} || {np.min(stats):.0f}, {np.quantile(stats, 0.25):.0f}, {np.quantile(stats, 0.5):.0f}, {np.quantile(stats, 0.75):.0f}, {np.max(stats):.0f}")
        stats = [int_variable_counts[instance] for instance in int_variable_counts]
        print(f"Integer variable stats : {np.mean(stats):.0f} || {np.min(stats):.0f}, {np.quantile(stats, 0.25):.0f}, {np.quantile(stats, 0.5):.0f}, {np.quantile(stats, 0.75):.0f}, {np.max(stats):.0f}")
        stats = [constraint_counts[instance] for instance in constraint_counts]
        print(f"Constraint stats       : {np.mean(stats):.0f} || {np.min(stats):.0f}, {np.quantile(stats, 0.25):.0f}, {np.quantile(stats, 0.5):.0f}, {np.quantile(stats, 0.75):.0f}, {np.max(stats):.0f}")
        stats = [nz_counts[instance] for instance in nz_counts]
        print(f"Nonzero stats          : {np.mean(stats):.0f} || {np.min(stats):.0f}, {np.quantile(stats, 0.25):.0f}, {np.quantile(stats, 0.5):.0f}, {np.quantile(stats, 0.75):.0f}, {np.max(stats):.0f}")
        stats = [nonlinear_nz_counts[instance] for instance in nonlinear_nz_counts]
        print(f"Nonlinear nonzero stats: {np.mean(stats):.0f} || {np.min(stats):.0f}, {np.quantile(stats, 0.25):.0f}, {np.quantile(stats, 0.5):.0f}, {np.quantile(stats, 0.75):.0f}, {np.max(stats):.0f}")
        all_stats = {}
        all_stats["ALL_nprob"] = [len(INSTANCES)]
        all_stats["ALL_ncons".format(solver.name)] = list(constraint_counts.values())
        all_stats["ALL_nvar".format(solver.name)] = list(variable_counts.values())
        all_stats["ALL_nintvar".format(solver.name)] = list(int_variable_counts.values())
        all_stats["ALL_nz".format(solver.name)] = list(nz_counts.values())
        all_stats["ALL_nlnz".format(solver.name)] = list(nonlinear_nz_counts.values())
        for solver in SOLVERS:
            all_stats["{}_nprob".format(solver.name)] = [len(INSTANCES) - solver.fails["total"]]
            all_stats["{}_ncons".format(solver.name)] = solved_instance_cons_counts[solver]
            all_stats["{}_nvar".format(solver.name)] = solved_instance_var_counts[solver]
            all_stats["{}_nintvar".format(solver.name)] = solved_instance_int_var_counts[solver]
            all_stats["{}_nz".format(solver.name)] = solved_instance_nz_counts[solver]
            all_stats["{}_nlnz".format(solver.name)] = solved_instance_nlnz_counts[solver]
        instance_size_plot(ARGS, all_stats, SOLVERS)

    FIG = plt.figure(figsize=(4, 4))
    RHO_TIME_PLOT = FIG.add_subplot(1, 1, 1)
    sorted_solvers = sorted(SOLVERS, key=lambda x: geometric_means[x.name])
    solver_positions = [next(i for i, s in enumerate(SOLVERS) if s.name == solver.name) for solver in sorted_solvers]

    if ARGS.primal:
        for i in range(len(sorted_solvers)):
            RHO_TIME_PLOT.plot(TIME, RHO_TIME[solver_positions[i]], color=COLORS[solver_positions[i]], linewidth=2.5,
                            mec=COLORS[solver_positions[i]], linestyle=STYLES[solver_positions[i]])
        for i in range(len(sorted_solvers)):
            RHO_TIME_PLOT.scatter(ARGS.failtime, RHO_TIME[solver_positions[i]][-1], s=300, facecolors='none', edgecolors=COLORS[solver_positions[i]], linestyle=STYLES[solver_positions[i]])
    else:
        for i in range(len(sorted_solvers)):
            RHO_TIME_PLOT.plot(TIME, RHO_TIME[solver_positions[i]], color=COLORS[solver_positions[i]], linewidth=2.5,
                            mec=COLORS[solver_positions[i]], linestyle=STYLES[solver_positions[i]])
            
    plt.gca().yaxis.grid(True)
    plt.xscale('log')
    # max_yaxis_value = min(100, 20 * (np.ceil(5 * max_solved_instance_count / len(INSTANCES)) + 1))
    if ARGS.time_factor:
        plt.axis([1, 1000, ARGS.ylo, ARGS.yup])
        plt.xlabel('Time Factor', fontsize=LABELS_SIZE, family='serif')
    else:
        plt.axis([1, ARGS.failtime, ARGS.ylo, ARGS.yup])
        plt.xlabel('Time [s]', fontsize=LABELS_SIZE, family='serif')
    plt.ylabel('Percent of models solved', fontsize=LABELS_SIZE, family='serif')
    plt.tick_params(axis='both', labelsize=TICKS_SIZE)
    RHO_TIME_PLOT.xaxis.set_major_formatter(ScalarFormatter())
    RHO_TIME_PLOT.yaxis.set_major_formatter(ScalarFormatter())
    xticks = [1]
    if ARGS.time_factor:
        while 10 * xticks[-1] < 1000:
            xticks.append(10 * xticks[-1])
        xticks.append(1000)
    else:
        while 10 * xticks[-1] < ARGS.failtime:
            xticks.append(10 * xticks[-1])
        xticks.append(ARGS.failtime)
    RHO_TIME_PLOT.set_xticks(xticks)
    legend_labels = [f"{solver.name} ({geometric_means[solver.name]:.1f} s)" for solver in sorted_solvers]
    LEG = plt.legend(legend_labels, loc="upper left",
                     bbox_to_anchor=(0, 1), fontsize=LEGEND_SIZE,
                     ncol=2)

    LEG.get_frame().set_edgecolor('none')
    FIGURE = plt.gcf()
    FIGURE.set_size_inches(12, 12)
    if ARGS.gap:
        print("Saving plots...")
    else:
        print("Saving plot...")
    plt.savefig(ARGS.timefilename, format="png", dpi=600)
    plt.close()

    if not ARGS.gap:
        quit()
    
    FIG = plt.figure(figsize=(4, 4))
    RHO_GAP_PLOT = FIG.add_subplot(1, 1, 1)
    for i in range(len(SOLVERS)):
        RHO_GAP_PLOT.plot(GAP, RHO_GAP[i], color=COLORS[i], linewidth=2.5,
                        mec=COLORS[i], linestyle=STYLES[i])
    plt.gca().yaxis.grid(True)
    plt.axis([0, 10, 0, 101])
    plt.xlabel(f'Remaining gap [%] at {int(ARGS.failtime):d} s', fontsize=LABELS_SIZE)
    plt.ylabel('Percent of unsolved models (%)', fontsize=LABELS_SIZE, family='serif')
    plt.tick_params(axis='both', labelsize=TICKS_SIZE)
    RHO_GAP_PLOT.xaxis.set_major_formatter(ScalarFormatter())
    RHO_GAP_PLOT.yaxis.set_major_formatter(ScalarFormatter())
    RHO_GAP_PLOT.set_xticks([0, 25, 50, 75, 100])
    legend_labels = ARGS.names
    LEG = plt.legend(legend_labels, loc="upper left",
                     bbox_to_anchor=(0, 1), fontsize=LEGEND_SIZE,
                     ncol=2)

    LEG.get_frame().set_edgecolor('none')
    FIGURE = plt.gcf()
    FIGURE.set_size_inches(12, 12)
    plt.savefig(ARGS.gapfilename, format="png", dpi=600)
