import os

if __name__ == "__main__":
    trace_file = 'a241018errorsALAMO241018.nvs'
    cmd = "python almplot.py traces/{}".format(trace_file)
    os.system(cmd)
    os.system("mv traces/{}.png figures/".format(trace_file.split('.')[0]))