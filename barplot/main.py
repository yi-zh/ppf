import os

if __name__ == "__main__":
    args = 'traces/BARON traces/BARONnew --instancesize'
    cmd = "python app.py {}".format(args)
    os.system(cmd)
